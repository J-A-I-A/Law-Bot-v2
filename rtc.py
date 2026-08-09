import asyncio
import base64
import json
import requests
import os
import httpx
import fractions
import logging
import re
import urllib.parse
import av
import numpy as np
import websockets
from websockets.exceptions import ConnectionClosed
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse,Response, PlainTextResponse
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from aiortc.contrib.media import MediaRelay
from openai import AsyncOpenAI


load_dotenv()

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
MODEL_API_KEY = os.getenv("MODEL_API_KEY")
META_PHONE_ID = os.getenv("META_PHONE_NUMBER_ID")
META_ACCESS_TOKEN = os.getenv("META_ACCESS_TOKEN", "").strip()
MODEL = os.getenv("MODEL")
ELEVENLABS_AGENT_ID = os.getenv("ELEVENLABS_AGENT_ID")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY").strip()
# Annakay Voice ID
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID")
VERIFY_TOKEN = "audiostreamtests"

app = FastAPI()
logger = logging.getLogger(__name__)

# TODO: 
# deepseek_client = AsyncOpenAI(
#     api_key=DEEPSEEK_API_KEY,
#     base_url="https://api.deepseek.com"
# )

import base64
import json
import asyncio
import urllib.parse
import websockets
import numpy as np

async def process_incoming_audio_track(track, api_key: str):
    """
    Streams WebRTC caller audio frames to ElevenLabs Scribe Realtime STT.
    Includes diagnostic logging for frame reception and audio energy.
    """
    if not api_key:
        logger.error("[STT Error] ELEVENLABS_API_KEY is missing or empty.")
        return

    stt_params = urllib.parse.urlencode({
        "model_id": "scribe_v2_realtime",
        "sample_rate": "16000",
        "audio_format": "pcm_16000"
    })
    stt_url = f"wss://api.elevenlabs.io/v1/speech-to-text/realtime?{stt_params}"
    extra_headers = {"xi-api-key": api_key}

    logger.info("[STT] Opening WebSocket connection to ElevenLabs Scribe...")

    try:
        async with websockets.connect(stt_url, additional_headers=extra_headers) as stt_ws:
            print("\n==========================================================")
            print(">>> [STT CONNECTED SUCCESSFULLY] Speak into your phone! <<<")
            print("==========================================================\n")

            async def send_audio_frames():
                frame_count = 0
                try:
                    while True:
                        # Fetch frame with 2s timeout to log if WebRTC audio is stuck
                        try:
                            frame = await asyncio.wait_for(track.recv(), timeout=2.0)
                        except asyncio.TimeoutError:
                            print("[STT WebRTC] Waiting for incoming audio frames from phone...")
                            continue

                        # Convert frame to numpy int16
                        pcm_array = frame.to_ndarray()

                        # Ensure 1D array
                        if pcm_array.ndim > 1:
                            pcm_array = pcm_array[0]
                        pcm_array = np.ascontiguousarray(pcm_array.flatten(), dtype=np.int16)

                        # Downsample 48kHz -> 16kHz
                        resampled_pcm = pcm_array[::3]
                        
                        # Diagnostic: Calculate RMS volume level to verify audio is not silent
                        rms = float(np.sqrt(np.mean(resampled_pcm.astype(np.float32)**2))) if len(resampled_pcm) > 0 else 0
                        
                        audio_bytes = resampled_pcm.tobytes()
                        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

                        payload = {
                            "message_type": "input_audio_chunk",
                            "audio_base_64": audio_b64
                        }
                        await stt_ws.send(json.dumps(payload))

                        frame_count += 1
                        if frame_count % 100 == 0:
                            print(f"[STT Active] Streamed {frame_count} frames (~2s) | Audio RMS Level: {rms:.1f}")

                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    logger.error(f"[STT Outbound Error]: {e}")

            async def listen_for_transcripts():
                try:
                    async for message in stt_ws:
                        data = json.loads(message)
                        msg_type = data.get("message_type")

                        if msg_type == "session_started":
                            print(f"\n[STT SESSION ACTIVE]: Session ID = {data.get('session_id')}\n")
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
                except Exception as e:
                    logger.error(f"[STT Inbound Error]: {e}")

            await asyncio.gather(send_audio_frames(), listen_for_transcripts())

    except websockets.exceptions.InvalidStatus as err:
        logger.error(f"[STT Reject] HTTP Status: {err.response.status_code}")
    except websockets.exceptions.ConnectionClosedOK:
        logger.info("[STT WebSocket] Connection closed cleanly.")
    except Exception as e:
        logger.error(f"[STT Exception]: {e}")

async def play_elevenlabs_greeting(voice_id: str, api_key: str, audio_queue: asyncio.Queue):
    """
    Triggers ElevenLabs TTS for the greeting message and queues 20ms PCM audio chunks 
    into audio_queue for WebRTC playback.
    """
    print("[ElevenLabs] Requesting initial greeting audio...")
    
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream?output_format=pcm_48000"
    
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json"
    }
    
    payload = {
        "text": "Hello! Thank you for calling. How can I help you today?",
        "model_id": "eleven_turbo_v2_5"
    }

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            async with client.stream("POST", url, headers=headers, json=payload) as response:
                if response.status_code != 200:
                    body = await response.aread()
                    logger.error(f"[ElevenLabs Error] {response.status_code}: {body}")
                    return
               
                frame_size = 1920
                buffer = bytearray()

                async for chunk in response.aiter_bytes():
                    buffer.extend(chunk)
                    while len(buffer) >= frame_size:
                        frame_bytes = bytes(buffer[:frame_size])
                        buffer = buffer[frame_size:]
                        await audio_queue.put(frame_bytes)

                if len(buffer) > 0:
                    padded = bytes(buffer) + b'\x00' * (frame_size - len(buffer))
                    await audio_queue.put(padded)

        logger.info("[ElevenLabs] Greeting audio queued successfully!")

    except Exception as e:
        logger.error(f"[ElevenLabs Error] Streaming failed: {e}")

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

    pc = RTCPeerConnection()
    audio_queue = asyncio.Queue()
    inbound_audio_track = None
    stt_task_started = False  # Lock flag to prevent duplicate connections

    @pc.on("track")
    def on_track(track):
        nonlocal inbound_audio_track
        if track.kind == "audio":
            logger.info("[WebRTC] Inbound caller audio track captured.")
            inbound_audio_track = track

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

    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(meta_url, headers=headers, json=accept_payload)
        logger.info(f"[Meta Accept Response]: {resp.status_code} - {resp.text}")

        if resp.status_code == 200 and not stt_task_started:
            stt_task_started = True  # Set lock
            logger.info("[Call Answered] Starting TTS Greeting & Realtime STT...")

            # Outbound greeting
            asyncio.create_task(
                play_elevenlabs_greeting(
                    voice_id=ELEVENLABS_VOICE_ID,
                    api_key=ELEVENLABS_API_KEY,
                    audio_queue=audio_queue
                )
            )

            # Inbound STT (single task)
            if inbound_audio_track:
                asyncio.create_task(
                    process_incoming_audio_track(
                        track=inbound_audio_track,
                        api_key=ELEVENLABS_API_KEY
                    )
                )

    await asyncio.Event().wait()

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
        asyncio.create_task(process_call_session(payload, meta_sdp_offer))
        return JSONResponse(status_code=200, content={"status": "processing"})

    return JSONResponse(status_code=200, content={"status": "ignored"})

class CustomAudioStreamTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(self, audio_queue):
        super().__init__()
        self.audio_queue = audio_queue
        self.pts = 0
        self.sample_rate = 48000
        self.samples_per_frame = 960  # 20ms frame at 48kHz

    async def recv(self):
        try:
            raw_pcm = self.audio_queue.get_nowait()
        except asyncio.QueueEmpty:
            raw_pcm = np.zeros(self.samples_per_frame, dtype=np.int16).tobytes()

        # Convert raw PCM16 bytes into numpy array
        audio_data = np.frombuffer(raw_pcm, dtype=np.int16).reshape(1, -1)

        # Build PyAV AudioFrame
        frame = av.AudioFrame.from_ndarray(audio_data, format="s16", layout="mono")
        frame.sample_rate = self.sample_rate
        frame.pts = self.pts
        frame.time_base = fractions.Fraction(1, self.sample_rate)

        self.pts += self.samples_per_frame
        await asyncio.sleep(0.02)
        return frame

async def stream_elevenlabs_tts(text_stream, audio_output_queue):
    tts_url = (
        f"wss://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}/stream-input"
        f"?model_id=eleven_turbo_v2_5&output_format=pcm_16000"
    )
    header = {"xi-api-key": ELEVENLABS_API_KEY}
    async with websockets.connect(tts_url,additional_headers=headers) as tts_ws:
        await tts_ws.send(json.dumps({
            "text": " ",
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.8}
        }))

        async def send_text_chunks():
            async for text_chunk in text_stream:
                if text_chunk:
                    await tts_ws.send(json.dumps({
                        "text": text_chunk,
                        "try_trigger_generation": True
                    }))
        
        async def receive_audio_chunks():
            async for message in tts_ws:
                data = json.loads(message)
                if data.get("audio"):
                    audio_bytes = base64.b64decode(data["audio"])
                    await audio_output_queue.put(audio_bytes)
        await asyncio.gather(send_text_chunks(), receive_audio_chunks())

async def process_deepseek_llm(transcript, audio_output_queue):
    print(f"\n[Whatsapp User]: {transcript}")
    print(f"[Deepseek Response]: ", end="", flush=True)

    # revisit llm transcript
    response = await deepseek_client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a concise voice phone assistant. Keep responses under 2 sentences and conversational."},
            {"role": "user", "content": transcript}
        ],
        stream=True
    )

    async def text_generator():
        async for chunk in response:
            delta = chunk.choices[0].delta.content
            if delta:
                print(delta, end="", flush=True)
                yield delta
    await stream_elevenlabs_tts(text_generator(), audio_output_queue)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)