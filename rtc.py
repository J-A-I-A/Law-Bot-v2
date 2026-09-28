import asyncio
import base64
import fractions
import json
import logging
import os
import re
import urllib.parse
import av
import httpx
import numpy as np
import requests
import websocket
import websockets
from bot import Law_bot
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack, RTCConfiguration, RTCIceServer
from aiortc.contrib.media import MediaRelay
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, PlainTextResponse
from websockets.exceptions import ConnectionClosed

load_dotenv()
active_call_tasks = set()

ELEVENLABS_API_KEY = (os.getenv("ELEVENLABS_API_KEY") or "").strip()
ELEVENLABS_AGENT_ID = (os.getenv("ELEVENLABS_AGENT_ID") or "").strip()
ELEVENLABS_VOICE_ID = (os.getenv("ELEVENLABS_VOICE_ID") or "").strip()
META_PHONE_ID = (os.getenv("META_PHONE_NUMBER_ID") or "").strip()
META_ACCESS_TOKEN = (os.getenv("META_ACCESS_TOKEN") or "").strip()
VERIFY_TOKEN = "audiostreamtests"

ELEVENLABS_STT_URL = "wss://api.elevenlabs.io/v1/speech-to-text/realtime"
ELEVENLABS_TTS_BASE_URL = "wss://api.elevenlabs.io/v1/text-to-speech"
ELEVENLABS_TTS_MODEL_ID = "eleven_turbo_v2_5"
ELEVENLABS_TTS_VOICE_SETTINGS = {"stability": 0.6, "similarity_boost": 0.1, "speed": 1.0}

STUN_URL = os.getenv("STUN_URL", "").strip()
TURN_URL = os.getenv("TURN_URL", "").strip()
TURN_USERNAME = os.getenv("TURN_USERNAME", "").strip()
TURN_CREDENTIAL = os.getenv("TURN_CREDENTIAL", "").strip()

app = FastAPI()
logger = logging.getLogger(__name__)


async def process_incoming_audio_track(track, ELEVENLABS_API_KEY: str, audio_output_queue):
    if not ELEVENLABS_API_KEY:
        logger.error("[Pipeline Error] ELEVENLABS_API_KEY is missing.")
        return

    stt_params = urllib.parse.urlencode({
        "model_id": "scribe_v2_realtime",
        "sample_rate": "16000",
        "audio_format": "pcm_16000",
        "commit_strategy": "vad",
        "vad_threshold": "0.6",
        "vad_silence_threshold_secs": "0.4",
        "min_speech_duration_ms": "150",
        "min_silence_duration_ms": "400"

    })
    stt_url = f"wss://api.elevenlabs.io/v1/speech-to-text/realtime?{stt_params}"
    extra_headers = {"xi-api-key": ELEVENLABS_API_KEY}

    try:
        async with (
            websockets.connect(stt_url, additional_headers=extra_headers) as stt_ws
            ):
            print("\n==========================================================")
            print(">>> [PIPELINE CONNECTED SUCCESSFULLY] Speak into your phone! <<<")
            print("==========================================================\n")

            async def send_audio_frames():
                frame_count = 0
                resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
                try:
                    while True:
                        # Fetch frame with 2s timeout to log if WebRTC audio is stuck
                        try:
                            frame = await asyncio.wait_for(track.recv(), timeout=2.0)
                        except asyncio.TimeoutError:
                            print("[STT WebRTC] Waiting for incoming audio frames from phone...")
                            continue

                        if frame_count == 0:
                            logger.info(
                                "[STT Audio Format] format=%s rate=%s layout=%s samples=%s",
                                frame.format.name,
                                frame.sample_rate,
                                frame.layout.name,
                                frame.samples,
                            )

                        for resampled_frame in resampler.resample(frame):
                            pcm_array = np.ascontiguousarray(
                                resampled_frame.to_ndarray().reshape(-1),
                                dtype=np.int16,
                            )
                            rms = (
                                float(np.sqrt(np.mean(pcm_array.astype(np.float32) ** 2)))
                                if len(pcm_array) > 0
                                else 0
                            )
                            payload = {
                                "message_type": "input_audio_chunk",
                                "audio_base_64": base64.b64encode(pcm_array.tobytes()).decode("utf-8"),
                            }
                            await stt_ws.send(json.dumps(payload))

                        frame_count += 1
                        if frame_count % 100 == 0:
                            print(f"[STT Active] Streamed {frame_count} frames (~2s) | Audio RMS Level: {rms:.1f}")

                except asyncio.CancelledError:
                    raise
                except ConnectionClosed:
                    logger.info("[STT Outbound] Websocket closed while streaming audio.")
                except Exception as e:
                    logger.error(f"[STT Outbound Error] {type(e).__name__}: {e!r}")

            async def listen_for_transcripts(stt_ws, audio_output_queue):
                try:
                    async for message in stt_ws:
                        data = json.loads(message)
                        msg_type = data.get("message_type")

                        if msg_type == "session_started":
                            print(f"\n[STT SESSION ACTIVE]: Session ID = {data.get('session_id')}\n")
                            continue

                        if msg_type == "committed_transcript":
                            text = data.get("text", "").strip()
                            if text:
                                print(f"\n[USER STT]: {text}\n")
                                asyncio.create_task(send_to_llm(text, audio_output_queue))
                            else:
                                logger.info("[STT Committed Transcript] Received empty transcript.")
                            continue                               
                            
                        # Print partial, final, or committed transcripts
                        text = data.get("text", "").strip()
                        if text:
                            tag = msg_type.upper().replace("_TRANSCRIPT", "") if msg_type else "TRANSCRIPT"
                            print(f"\n==============================================")
                            print(f"  [{tag}]: {text}")
                            print(f"==============================================\n")
                        elif msg_type not in ("session_started", "input_audio_chunk"):
                            print(f"[STT Inbound Event]: {data}")

                except asyncio.CancelledError:
                    pass
                except ConnectionClosed:
                    logger.info("[STT Inbound] Websocket closed.")
                except Exception as e:
                    logger.error(f"[STT Inbound Error] {type(e).__name__}: {e!r}")

            sender_task = asyncio.create_task(send_audio_frames())
            listener_task = asyncio.create_task(
                listen_for_transcripts(stt_ws, audio_output_queue)
            )
            done, pending = await asyncio.wait(
                (sender_task, listener_task),
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            await asyncio.gather(*done, return_exceptions=True)

    except websockets.exceptions.InvalidStatus as err:
        logger.error(f"[STT Reject] HTTP Status: {err.response.status_code}")
    except websockets.exceptions.ConnectionClosedOK:
        logger.info("[STT WebSocket] Connection closed cleanly.")
    except Exception as e:
        logger.error(f"[STT Exception]: {e}")

def normalize_meta_sdp_answer(answer_sdp: str) -> str:
    if not answer_sdp:
        return answer_sdp
    normalized_lines = []
    seen_sha256_fingerprint = False

    for raw_lines in answer_sdp.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = raw_lines.strip()
        if not line:
            continue
        if line.lower().startswith("a=fingerprint:sha-256"):
            if not seen_sha256_fingerprint:
                parts = line.split(" ", 1)
                fp_val = parts[1] if len(parts) > 1 else ""
                normalized_lines.append(f"a=fingerprint:sha-256 {fp_val}".strip())
                seen_sha256_fingerprint = True
            continue
        if line.startswith("a=fingerprint:sha-384") or line.startswith("a=fingerprint:sha-512"):
            continue
        if line.startswith("a=setup:"):
            normalized_lines.append("a=setup:passive")
            continue
        if line.startswith("a=msid:") or line.startswith("a=ssrc:") or "telephone-event" in line:
            continue
        normalized_lines.append(line)
    return "\r\n".join(normalized_lines) + "\r\n"

# --- SESSION CONTROLLER ---
async def process_call_session(payload: dict, meta_sdp_offer: str):
    entry = payload.get("entry", [{}])[0]
    change = entry.get("changes", [{}])[0]
    value = change.get("value", {})
    calls = value.get("calls", [{}])
    
    if not calls:
        return
        
    call_id = calls[0].get("id")
    print("\n========== META SDP OFFER ==========")
    print(meta_sdp_offer, flush=True)
    print("======== END META SDP OFFER ========\n", flush=True)
# STUN and TURN server configuration for WebRTC
    config = RTCConfiguration(
        iceServers=[
            RTCIceServer(urls = STUN_URL),
            RTCIceServer(
                urls = TURN_URL,
                username= TURN_USERNAME,
                credential= TURN_CREDENTIAL 
            )
        ]
    )

    pc = RTCPeerConnection(configuration=config)
    audio_queue = asyncio.Queue()
    call_active_event = asyncio.Event()
    pipeline_started = False

    @pc.on("iceconnectionstatechange")
    def on_ice_connection_state_change():
        state = pc.iceConnectionState
        logger.info("[WebRTC] ICE connection state changed: %s", state)

        if state in ("failed", "closed"):
            call_active_event.set()

    @pc.on("track")
    def on_track(track):
        nonlocal pipeline_started

        if track.kind != "audio" or pipeline_started:
            return

        pipeline_started = True
        logger.info("[WebRTC] Inbound caller audio track captured.")

        pipeline_task = asyncio.create_task(
            process_incoming_audio_track(track, ELEVENLABS_API_KEY, audio_queue)
        )
        active_call_tasks.add(pipeline_task)
        pipeline_task.add_done_callback(active_call_tasks.discard)

    # Set Remote Offer and Audio Transceiver setup
    offer = RTCSessionDescription(sdp=meta_sdp_offer, type="offer")
    await pc.setRemoteDescription(offer)

    local_track = CustomAudioStreamTrack(audio_queue)
    audio_transceiver = None

    for t in pc.getTransceivers():
        if t.kind == "audio":
            audio_transceiver = t
            break

    if audio_transceiver:
        audio_transceiver.sender.replaceTrack(local_track)
        audio_transceiver.direction = "sendrecv"
    else:
        pc.addTrack(local_track)

    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    cleaned_sdp = normalize_meta_sdp_answer(pc.localDescription.sdp)

    accept_payload = {
        "messaging_product": "whatsapp",
        "call_id": call_id,
        "action": "accept",
        "session": {
            "sdp_type": "answer",
            "sdp": cleaned_sdp
        }
    }

    meta_url = f"https://graph.facebook.com/v20.0/{META_PHONE_ID}/calls"
    headers = {
        "Authorization": f"Bearer {META_ACCESS_TOKEN}",
        "Content-Type": "application/json"
    }

    async def cleanup_call():
        try:
            await call_active_event.wait()
        finally:
            logger.info("[Call Ended] Closing WebRTC peer connection.")
            await pc.close()

    cleanup_task = asyncio.create_task(cleanup_call())
    active_call_tasks.add(cleanup_task)
    cleanup_task.add_done_callback(active_call_tasks.discard)

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                meta_url,
                headers=headers,
                json=accept_payload,
            )

        logger.info(
            "[Meta Accept Response]: %s - %s",
            resp.status_code,
            resp.text,
        )

        if resp.status_code != 200:
            call_active_event.set()
            raise RuntimeError(
                f"Meta rejected call: {resp.status_code} {resp.text}"
            )

        return cleaned_sdp

    except Exception:
        call_active_event.set()
        raise

@app.get("/whatsapp-call")
async def verify_webhook(request: Request):
    """Handles Meta's initial Webhook URL Verification challenge."""
    params = request.query_params
    mode = params.get("hub.mode")
    token = params.get("hub.verify_token")
    challenge = params.get("hub.challenge")

    if mode == "subscribe" and token == VERIFY_TOKEN:
        print("\n[Meta Webhook Verified Successfully!]")
        return PlainTextResponse(content=challenge, status_code=200)
    return Response(status_code=403)


@app.post("/whatsapp-call")
async def handle_meta_calling_webhook(request: Request):
    payload = await request.json()

    call_id = None
    meta_sdp_offer = None
    event_type = None

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
            meta_sdp_offer = session.get("sdp")

    except Exception as e:
        logger.error(f"Error parsing Meta payload: {e}")
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

    if event_type and event_type != "connect":
        logger.info("Received event '%s' for call %s", event_type, call_id)
        return JSONResponse(status_code=200, content={"status": "acknowledged"})
    

    if meta_sdp_offer:
        sdp_answer = await process_call_session(payload, meta_sdp_offer)

        # active_call_tasks.add(call_task)
        # call_task.add_done_callback(active_call_tasks.discard)
        # sdp_answer = await call_task
        return JSONResponse(
            status_code=200, content={"status": "accepted", "sdp_answer": sdp_answer}
        )

    return JSONResponse(status_code=200, content={"status": "ignored"})

class CustomAudioStreamTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(self, audio_queue):
        super().__init__()
        self.audio_queue = audio_queue
        self.pts = 0
        self.sample_rate = 48000
        self.samples_per_frame = 960  # 20ms frame at 48kHz
        self.frame_size_bytes = self.samples_per_frame * 2
        self.pending_audio = bytearray()
        self.audio_stream_ended = False
        self.audio_frames_sent = 0

    async def recv(self):
        loop = asyncio.get_running_loop()
        frame_deadline = loop.time() + 0.02

        while len(self.pending_audio) < self.frame_size_bytes:
            timeout = frame_deadline - loop.time()
            if timeout <= 0:
                break
            try:
                audio_chunk = await asyncio.wait_for(self.audio_queue.get(), timeout)
            except asyncio.TimeoutError:
                break

            if audio_chunk is None:
                self.audio_stream_ended = True
                break

            self.pending_audio.extend(audio_chunk)
            self.audio_stream_ended = False

        if len(self.pending_audio) >= self.frame_size_bytes:
            raw_pcm = bytes(self.pending_audio[:self.frame_size_bytes])
            del self.pending_audio[:self.frame_size_bytes]
        elif self.audio_stream_ended and self.pending_audio:
            raw_pcm = bytes(self.pending_audio).ljust(self.frame_size_bytes, b"\x00")
            self.pending_audio.clear()
            self.audio_stream_ended = False
        else:
            raw_pcm = bytes(self.frame_size_bytes)

        remaining_frame_time = frame_deadline - loop.time()
        if remaining_frame_time > 0:
            await asyncio.sleep(remaining_frame_time)

        # Convert raw PCM16 bytes into numpy array
        audio_data = np.frombuffer(raw_pcm, dtype=np.int16).reshape(1, -1)

        # Build PyAV AudioFrame
        frame = av.AudioFrame.from_ndarray(audio_data, format="s16", layout="mono")
        frame.sample_rate = self.sample_rate
        frame.pts = self.pts
        frame.time_base = fractions.Fraction(1, self.sample_rate)

        self.pts += self.samples_per_frame
        if self.audio_frames_sent == 0 and np.any(audio_data):
            logger.info("[TTS Audio] First non-silent PCM frame sent to WebRTC.")
        if np.any(audio_data):
            self.audio_frames_sent += 1
        return frame

async def stream_elevenlabs_tts(text_stream, audio_output_queue):
    tts_url = (
        f"{ELEVENLABS_TTS_BASE_URL}/{ELEVENLABS_VOICE_ID}/stream-input"
        f"?model_id={ELEVENLABS_TTS_MODEL_ID}&output_format=pcm_48000"
    )
    header = {"xi-api-key": ELEVENLABS_API_KEY}

    async with websockets.connect(tts_url, additional_headers=header) as tts_ws:
        await tts_ws.send(
            json.dumps({
                "text": " ",
                "voice_settings": ELEVENLABS_TTS_VOICE_SETTINGS,
                "language_code": "en",
            })
        )

        async def send_text_chunks():
            text_buffer = ""

            async for text_chunk in text_stream:
                if not text_chunk:
                    continue

                text_buffer += text_chunk
                while True:
                    boundary = re.search(r"[.!?;:]\s+", text_buffer)
                    if boundary:
                        split_at = boundary.end()
                    elif len(text_buffer) >= 120:
                        split_at = text_buffer.rfind(" ", 0, 120)
                        if split_at <= 0:
                            split_at = 120
                    else:
                        break

                    phrase = text_buffer[:split_at]
                    text_buffer = text_buffer[split_at:]
                    await tts_ws.send(
                        json.dumps({
                            "text": phrase,
                            "try_trigger_generation": True,
                        })
                    )

            if text_buffer.strip():
                await tts_ws.send(
                    json.dumps({
                        "text": text_buffer,
                        "try_trigger_generation": True,
                    })
                )
            await tts_ws.send(json.dumps({"text": ""}))
        # revisit TODO
        async def receive_audio_chunks():
            try:
                async for message in tts_ws:
                    data = json.loads(message)

                    if data.get("audio"):
                        await audio_output_queue.put(base64.b64decode(data["audio"]))

                    if data.get("isFinal"):
                        break
            finally:
                await audio_output_queue.put(None)

        await asyncio.gather(send_text_chunks(), receive_audio_chunks())

async def send_to_llm(transcript: str, audio_output_queue):
    print(f"\n[Whatsapp User]: {transcript}")
    print("[LLM Response]: ", end="", flush=True)

    try:
        loop = asyncio.get_running_loop()
        text_queue = asyncio.Queue()

        async def text_generator():
            while True:
                text_chunk = await text_queue.get()
                if text_chunk is None:
                    break
                yield text_chunk

        def on_delta(text_chunk: str):
            print(text_chunk, end="", flush=True)
            loop.call_soon_threadsafe(text_queue.put_nowait, text_chunk)

        tts_task = asyncio.create_task(
            stream_elevenlabs_tts(text_generator(), audio_output_queue)
        )
        try:
            await asyncio.to_thread(
                Law_bot,
                previous_message=[],
                question=transcript,
                on_delta=on_delta,
            )
        finally:
            await text_queue.put(None)
            await tts_task
        print(flush=True)

    except Exception as e:
        logger.error(f"[LLM Error]: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)