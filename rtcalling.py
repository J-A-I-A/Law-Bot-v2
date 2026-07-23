import asyncio
import base64
import fractions
import io
import json
import logging
import av
import httpx
import websockets
from aiortc import RTCConfiguration, RTCIceServer, RTCSessionDescription, RTCPeerConnection, MediaStreamTrack
from aiortc.mediastreams import MediaStreamError
from fastapi import FastAPI, Query, Request
from starlette.responses import JSONResponse

logger = logging.getLogger("uvicorn")
app = FastAPI()

CALL_SESSIONS: dict[str, dict] = {}

META_ACCESS_TOKEN = 

ELEVENLABS_AGENT_ID = 
ELEVENLABS_API_KEY = 
VERIFY_TOKEN = 
MAX_OUTBOUND_QUEUE_SIZE = 100
CALL_SETUP_TIMEOUT_SECONDS = 15
WEBSOCKET_READ_TIMEOUT_SECONDS = 30

@app.get("/whatsapp-call")
async def verify_meta_webhook(
    mode: str = Query(None, alias="hub.mode"),
    token: str = Query(None, alias="hub.verify_token"),
    challenge: str = Query(None, alias="hub.challenge"),
):
    if mode == "subscribe" and token == VERIFY_TOKEN:
        logger.info("Webhook verified successfully.")
        return int(challenge) if challenge.isdigit() else challenge
    return {"error": "Verification failed. Invalid token or mode."}, 403
audio_bridge_resampler = av.AudioResampler(
    format='s16',
    layout='mono',
    rate=16000,
)
audio_frame_resampler = av.AudioResampler(
    format='s16',
    layout='mono',
    rate=16000,
)


def decode_elevenlabs_audio_frames(raw_bytes: bytes):
    if not raw_bytes:
        return []

    try:
        with av.open(io.BytesIO(raw_bytes)) as container:
            frames = [frame for frame in container.decode(audio=0)]
            if frames:
                logger.debug("Decoded %d container audio frames from ElevenLabs payload.", len(frames))
                return frames
    except Exception as exc:
        logger.debug("ElevenLabs payload is not a container or decoder failed: %s", exc)

    # Fall back to raw 16kHz PCM if container decode fails.
    sample_count = len(raw_bytes) // 2
    if sample_count <= 0:
        return []

    raw_bytes = raw_bytes[: sample_count * 2]
    frame = av.AudioFrame(format="s16", layout="mono", samples=sample_count)
    frame.sample_rate = 16000
    frame.planes[0].update(raw_bytes)
    logger.debug("Interpreting ElevenLabs payload as raw PCM16: samples=%d", sample_count)
    return [frame]


def normalize_audio_frame_to_pcm16_mono(frame: av.AudioFrame) -> bytes:
    if frame.format.name == "s16" and frame.layout.name == "mono" and frame.sample_rate == 16000:
        return frame.to_ndarray().tobytes()

    normalized_frames = audio_frame_resampler.resample(frame)
    raw_bytes = b"".join(f.to_ndarray().tobytes() for f in normalized_frames)
    logger.debug(
        "Normalized ElevenLabs frame: orig_rate=%s orig_layout=%s output_bytes=%d",
        frame.sample_rate,
        frame.layout.name if frame.layout else "unknown",
        len(raw_bytes),
    )
    return raw_bytes


def split_pcm16_into_20ms_chunks(raw_bytes: bytes) -> list[bytes]:
    frame_size = max(1, int(16000 * 0.02 * 2))
    chunks = []
    idx = 0
    while idx + frame_size <= len(raw_bytes):
        chunks.append(raw_bytes[idx:idx + frame_size])
        idx += frame_size
    return chunks


async def close_call_session(call_id: str, reason: str = "unknown"):
    session = CALL_SESSIONS.pop(call_id, None)
    if not session:
        return

    logger.info("Closing call session %s (%s)", call_id, reason)

    pc = session.get("pc")
    outbound_track = session.get("outbound_track")
    el_websocket = session.get("el_websocket")
    audio_task = session.get("audio_task")
    forward_task = session.get("forward_task")

    for task in (audio_task, forward_task):
        if task is not None and not task.done():
            task.cancel()

    if outbound_track is not None:
        try:
            outbound_track.close()
        except Exception:
            pass

    if pc is not None:
        try:
            await pc.close()
        except Exception:
            pass

    if el_websocket is not None:
        try:
            await el_websocket.close()
        except Exception:
            pass


async def stream_user_audio_to_elevenlabs(track, el_websocket, call_id: str):
    while True:
        try:
            frame = await asyncio.wait_for(track.recv(), timeout=WEBSOCKET_READ_TIMEOUT_SECONDS)
            resampled_frames = audio_bridge_resampler.resample(frame)
            for resampled_frame in resampled_frames:
                raw_pcm_bytes = resampled_frame.to_ndarray().tobytes()
                base64_audio = base64.b64encode(raw_pcm_bytes).decode("utf-8")
                await asyncio.wait_for(
                    el_websocket.send(json.dumps({
                        "user_audio_chunk": base64_audio,
                    })),
                    timeout=WEBSOCKET_READ_TIMEOUT_SECONDS,
                )
        except asyncio.TimeoutError:
            logger.warning("Audio forwarding timed out for call %s; closing bridge.", call_id)
            break
        except asyncio.CancelledError:
            raise
        except MediaStreamError:
            logger.info("Meta WebRTC track closed for call %s.", call_id)
            break
        except websockets.exceptions.ConnectionClosed:
            logger.info("ElevenLabs WebSocket connection closed for call %s.", call_id)
            break
        except Exception as exc:
            logger.exception("Bridge pipeline exception encountered for call %s: %s", call_id, exc)
            break

class ElevenLabsToWebRTCTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(self):
        super().__init__()
        self.queue = asyncio.Queue(maxsize=MAX_OUTBOUND_QUEUE_SIZE)
        self.outbound_resampler = av.AudioResampler(
            format="s16",
            layout="mono",
            rate=48000,
        )
        self.pts = 0
        self._closed = False

    def close(self):
        self._closed = True
        while not self.queue.empty():
            self.queue.get_nowait()

    async def add_pcm_chunk(self, raw_bytes_16k: bytes):
        if self._closed or not raw_bytes_16k:
            return

        input_frame = av.AudioFrame(format="s16", layout="mono", samples=len(raw_bytes_16k) // 2)
        input_frame.sample_rate = 16000
        input_frame.planes[0].update(raw_bytes_16k)

        resampled_frames = self.outbound_resampler.resample(input_frame)
        for resample in resampled_frames:
            if self.queue.full():
                self.queue.get_nowait()
            self.queue.put_nowait(resample)
            logger.debug(
                "Queued resampled ElevenLabs frame: samples=%s pts=%s time_base=%s",
                resample.samples,
                resample.pts,
                resample.time_base,
            )

    async def recv(self):
        if self._closed:
            raise MediaStreamError("Outbound track closed.")

        frame = await self.queue.get()
        frame.pts = self.pts
        frame.time_base = fractions.Fraction(1, 48000)
        self.pts += frame.samples
        logger.debug(
            "Transmitting frame to Meta: samples=%s pts=%s time_base=%s queue_size=%s",
            frame.samples,
            frame.pts,
            frame.time_base,
            self.queue.qsize(),
        )
        return frame

async def listen_to_elevenlabs_and_forward(el_websocket, outbound_track, call_id: str):
    try:
        while True:
            try:
                message = await asyncio.wait_for(el_websocket.recv(), timeout=WEBSOCKET_READ_TIMEOUT_SECONDS)
            except asyncio.TimeoutError:
                logger.warning("ElevenLabs websocket timed out for call %s; closing outbound track.", call_id)
                break
            except websockets.ConnectionClosed:
                break

            try:
                data = json.loads(message)
            except json.JSONDecodeError:
                logger.debug("Ignoring non-JSON ElevenLabs message: %s", message)
                continue

            audio_b64 = None
            if isinstance(data.get("audio"), str):
                audio_b64 = data["audio"]
            elif isinstance(data.get("audio"), dict):
                audio_b64 = data["audio"].get("audio_base_64") or data["audio"].get("audio_base64")
            elif isinstance(data.get("audio_event"), dict):
                audio_b64 = data["audio_event"].get("audio_base_64") or data["audio_event"].get("audio_base64")
            elif isinstance(data.get("data"), dict):
                audio_b64 = data["data"].get("audio_base_64") or data["data"].get("audio_base64")
            elif data.get("type") in ("audio", "response"):
                audio_b64 = data.get("audio_base_64") or data.get("audio_base64")

            if audio_b64:
                try:
                    payload_bytes = base64.b64decode(audio_b64)
                    logger.debug("Received ElevenLabs audio chunk (%d bytes)", len(payload_bytes))

                    for decoded_frame in decode_elevenlabs_audio_frames(payload_bytes):
                        normalized_pcm = normalize_audio_frame_to_pcm16_mono(decoded_frame)
                        logger.debug(
                            "Decoded ElevenLabs frame: samples=%s sample_rate=%s layout=%s normalized_bytes=%d",
                            decoded_frame.samples,
                            decoded_frame.sample_rate,
                            decoded_frame.layout.name if decoded_frame.layout else "unknown",
                            len(normalized_pcm),
                        )
                        for chunk in split_pcm16_into_20ms_chunks(normalized_pcm):
                            logger.debug("Forwarding 20ms chunk to outbound track: %d bytes", len(chunk))
                            await outbound_track.add_pcm_chunk(chunk)
                except Exception as exc:
                    logger.error("Failed to decode ElevenLabs audio chunk: %s", exc)
            else:
                logger.debug("Non-audio ElevenLabs payload: %s", json.dumps(data))
    except asyncio.CancelledError:
        raise
    except websockets.exceptions.ConnectionClosed:
        logger.info("ElevenLabs websocket closed for call %s.", call_id)
    except Exception as exc:
        logger.exception("Error receiving from ElevenLabs for call %s: %s", call_id, exc)
    finally:
        outbound_track.close()


@app.post("/whatsapp-call")
async def handle_meta_calling_webhook(request: Request):
    payload = await request.json()

    call_id = None
    meta_sdp_offer = None
    event_type = None
    session_type = None
    session_sdp = None

    try:
        entry = payload.get("entry", [{}])[0]
        change = entry.get("changes", [{}])[0]
        value = change.get("value", {})
        calls = value.get("calls", [])
        if calls:
            call_data = calls[0]
            call_id = call_data.get("id")
            event_type = call_data.get("event")  
            
            session = call_data.get("session", {})
            session_type = session.get("type")
            session_sdp = session.get("sdp")

            logger.debug(
                "Meta call_data payload: %s",
                json.dumps(call_data, indent=2)
            )

            if session_sdp:
                meta_sdp_offer = session_sdp
                if not session_type:
                    logger.info(
                        "Meta payload contains SDP but no session.type; assuming incoming offer on connect event."
                    )

        logger.info(
            "WhatsApp call webhook received: event_type=%s call_id=%s session_type=%s has_sdp=%s",
            event_type,
            call_id,
            session_type,
            bool(session_sdp),
        )

    except (KeyError, IndexError) as e:
        logger.error(f"Failed to parse Meta call payload structure: {e}")
        return JSONResponse(status_code=200, content={"status": "ignored", "reason": "Invalid payload structure"})
    except Exception as e:
        logger.error(f"Unexpected error parsing Meta call payload: {e}")
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

    if event_type and event_type != "connect":
        if call_id:
            await close_call_session(call_id, reason=f"meta-event:{event_type}")
        logger.info("Received call event '%s' for call %s; closing any active session.", event_type, call_id)
        return JSONResponse(status_code=200, content={"status": "acknowledged"})

    if not meta_sdp_offer:
        logger.warning("No valid meta_sdp_offer found in request payload. Skipping session processing.")
        return JSONResponse(status_code=200, content={"status": "ignored", "reason": "No SDP offer found"})

    try:
        await process_call_session(payload, meta_sdp_offer)
        return JSONResponse(status_code=200, content={"status": "success"})
    except Exception as e:
        logger.exception("Error processing call session: %s", e)
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

def normalize_meta_sdp_answer(answer_sdp: str) -> str:
    if not answer_sdp:
        return answer_sdp

    normalized_lines = []
    seen_sha256_fingerprint = False

    for raw_line in answer_sdp.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = raw_line.strip()
        if not line:
            continue

        if line.startswith("a=fingerprint:sha-256"):
            if not seen_sha256_fingerprint:
                normalized_lines.append(line)
                seen_sha256_fingerprint = True
            continue

        if line.startswith("a=fingerprint:sha-384") or line.startswith("a=fingerprint:sha-512"):
            continue

        if line.startswith("a=setup:actpass"):
            normalized_lines.append("a=setup:active")
            continue

        normalized_lines.append(line)

    normalized = "\n".join(normalized_lines).strip()

    if "a=sendrecv" not in normalized:
        normalized += "\na=sendrecv"

    if "a=rtcp-mux" not in normalized:
        normalized += "\na=rtcp-mux"

    return normalized

async def process_call_session(payload: dict, meta_sdp_offer: str):
    config = RTCConfiguration(
        iceServers=[RTCIceServer(urls=["stun:stun.l.google.com:19302"])],
    )
    pc = RTCPeerConnection(configuration=config)

    signed_url = f"wss://api.elevenlabs.io/v1/convai/conversation?agent_id={ELEVENLABS_AGENT_ID}"
    el_headers = {"xi-api-key": ELEVENLABS_API_KEY}

    call_id = payload["entry"][0]["changes"][0]["value"]["calls"][0]["id"]
    CALL_SESSIONS[call_id] = {"status": "connecting", "pc": pc}

    try:
        el_websocket = await asyncio.wait_for(
            websockets.connect(signed_url, additional_headers=el_headers),
            timeout=CALL_SETUP_TIMEOUT_SECONDS,
        )

        @pc.on("track")
        def on_track(track):
            if track.kind == "audio":
                logger.info("Successfully connected to WhatsApp inbound audio stream for call %s.", call_id)
                session = CALL_SESSIONS.get(call_id)
                if session is not None:
                    session["audio_task"] = asyncio.create_task(
                        stream_user_audio_to_elevenlabs(track, el_websocket, call_id)
                    )

        offer = RTCSessionDescription(sdp=meta_sdp_offer, type="offer")
        await asyncio.wait_for(pc.setRemoteDescription(offer), timeout=CALL_SETUP_TIMEOUT_SECONDS)

        outbound_track = ElevenLabsToWebRTCTrack()
        logger.debug("Created ElevenLabs outbound audio track and attaching to PC.")
        pc.addTrack(outbound_track)
        CALL_SESSIONS[call_id]["outbound_track"] = outbound_track
        CALL_SESSIONS[call_id]["el_websocket"] = el_websocket

        session = CALL_SESSIONS.get(call_id)
        if session is not None:
            session["forward_task"] = asyncio.create_task(
                listen_to_elevenlabs_and_forward(el_websocket, outbound_track, call_id)
            )

        answer = await asyncio.wait_for(pc.createAnswer(), timeout=CALL_SETUP_TIMEOUT_SECONDS)
        await asyncio.wait_for(pc.setLocalDescription(answer), timeout=CALL_SETUP_TIMEOUT_SECONDS)

        raw_sdp = pc.localDescription.sdp
        modified_sdp = normalize_meta_sdp_answer(raw_sdp)
        logger.info("Generated SDP answer for Meta: %s", modified_sdp)

        phone_number_id = payload["entry"][0]["changes"][0]["value"]["metadata"]["phone_number_id"]

        meta_call_url = f"https://graph.facebook.com/v25.0/{phone_number_id}/calls"
        meta_headers = {
            "Authorization": f"Bearer {META_ACCESS_TOKEN}",
            "Content-Type": "application/json",
        }
        accept_payload = {
            "messaging_product": "whatsapp",
            "call_id": call_id,
            "action": "accept",
            "session": {
                "sdp": modified_sdp,
                "sdp_type": "answer",
            },
        }
        async with httpx.AsyncClient() as client:
            res_accept = await asyncio.wait_for(
                client.post(meta_call_url, headers=meta_headers, json=accept_payload),
                timeout=CALL_SETUP_TIMEOUT_SECONDS,
            )
            logger.info("Meta accept response: %s - %s", res_accept.status_code, res_accept.text)

        if res_accept.status_code != 200:
            raise RuntimeError(
                f"Meta call accept failed: {res_accept.status_code} - {res_accept.text}"
            )

        CALL_SESSIONS[call_id]["status"] = "active"
        logger.info("Handshake patched successfully for call %s. Sending finalized SDP answer directly back to Meta.", call_id)

        return {
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type,
        }
    except asyncio.TimeoutError:
        if call_id in CALL_SESSIONS:
            CALL_SESSIONS[call_id]["status"] = "failed"
        logger.warning("Call setup timed out for call %s.", call_id)
        await close_call_session(call_id, reason="setup-timeout")
        raise
    except Exception:
        if call_id in CALL_SESSIONS:
            CALL_SESSIONS[call_id]["status"] = "failed"
        logger.exception("Call session failed for call %s.", call_id)
        await close_call_session(call_id, reason="exception")
        raise
    finally:
        if call_id in CALL_SESSIONS:
            CALL_SESSIONS[call_id].setdefault("status", "closed")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)