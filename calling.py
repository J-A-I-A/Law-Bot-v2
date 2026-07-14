"""WhatsApp Calling <-> Azure OpenAI Realtime (gpt-realtime) voice bridge.

This is an always-on async service (deployed separately from the Azure Functions
app, e.g. to Azure Container Apps). It gives voice callers the *same* legal tool
the WhatsApp chat bot uses (`get_info` -> `relevant_info`), backed by the
`gpt-realtime` model.

Flow for an inbound WhatsApp call
---------------------------------
1. The Functions webhook (`whatsapp.py`) receives a `calls` webhook event with a
   WebRTC SDP *offer* and POSTs it here to `/start_call`.
2. We create an ephemeral Realtime session (instructions + voice + tools), then
   relay WhatsApp's SDP offer straight to Azure's Realtime `/calls` endpoint and
   get back an SDP *answer* plus a `call_id`. Audio then flows peer-to-peer
   between Meta and Azure over WebRTC/OPUS -- this service never touches audio.
3. We hand that SDP answer back to WhatsApp via the Graph `/calls` endpoint
   (`pre_accept` then `accept`).
4. We open a server-side *control* WebSocket (`?call_id=...`) for the lifetime of
   the call. That socket is where `get_info` tool calls are serviced -- exactly
   like the chat bot -- by calling `relevant_info()` and returning the result.

The heavy media path (WebRTC/SRTP/OPUS) is handled entirely by Meta and Azure;
we only relay signaling (SDP) and drive tool calls over the control socket.
"""

import asyncio
import json
import logging
import os

import aiohttp
from aiohttp import web
from dotenv import load_dotenv

from relevant_info import relevant_info

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("calling")

# --- Azure OpenAI Realtime config -------------------------------------------
REALTIME_ENDPOINT = os.getenv("REALTIME_ENDPOINT", "").strip().rstrip("/")
REALTIME_API_KEY = os.getenv("REALTIME_API_KEY", "")
# Deployment name of the realtime model (strip quotes that may be in .env)
REALTIME_MODEL = os.getenv("REALTIME_MODEL", "gpt-realtime-2.1").strip().strip('"')
# Output voice for the model. One of: alloy, ash, ballad, coral, echo, sage,
# shimmer, verse, marin, cedar (availability depends on model version).
REALTIME_VOICE = os.getenv("REALTIME_VOICE", "alloy").strip()

_REALTIME_HOST = REALTIME_ENDPOINT.replace("https://", "").replace("http://", "")
CLIENT_SECRETS_URL = f"{REALTIME_ENDPOINT}/openai/v1/realtime/client_secrets"
CALLS_URL = f"{REALTIME_ENDPOINT}/openai/v1/realtime/calls"
WS_URL_TEMPLATE = f"wss://{_REALTIME_HOST}/openai/v1/realtime?call_id={{call_id}}"

# --- WhatsApp Cloud API config (shared with whatsapp.py) --------------------
WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN", "")
WHATSAPP_PHONE_ID = os.getenv("WHATAPP_PHONE_ID", "")  # NOTE: spelling matches .env
GRAPH_API_VERSION = "v25.0"
GRAPH_CALLS_URL = f"https://graph.facebook.com/{GRAPH_API_VERSION}/{WHATSAPP_PHONE_ID}/calls"

# Some WebRTC peers reject an SDP answer whose DTLS role is left as `actpass`.
# If the first live call fails to connect media, set SDP_FORCE_SETUP_ACTIVE=1.
SDP_FORCE_SETUP_ACTIVE = os.getenv("SDP_FORCE_SETUP_ACTIVE", "0") == "1"

HTTP_TIMEOUT = aiohttp.ClientTimeout(total=30)

# Registry of in-flight calls keyed by the WhatsApp (Meta) call id, so a
# `terminate` webhook can cancel the matching control-socket task.
ACTIVE_CALLS: dict[str, asyncio.Task] = {}

# Strong refs to background tasks so the event loop doesn't garbage-collect them
# while they run (asyncio only holds weak references to tasks).
_BG_TASKS: set[asyncio.Task] = set()


def _spawn(coro) -> None:
    task = asyncio.create_task(coro)
    _BG_TASKS.add(task)
    task.add_done_callback(_BG_TASKS.discard)


# --- Prompt + tool (mirrors the chat bot's `get_info`) ----------------------

VOICE_INSTRUCTIONS = """You are JAIA, a voice assistant that answers questions about Jamaican law over a phone call.

You are speaking out loud, so:
- Keep answers short, clear, and conversational. No markdown, headings, or bullet symbols.
- Do not read out long statutory text verbatim; summarize it plainly, then name the source.

For every legal question you MUST call the get_info tool to retrieve authoritative Jamaican law before answering, and cite the source (the act/section) the tool returns. Never invent a law or citation.

Retrieving the law takes a couple of seconds, and the caller only hears silence while it runs. So BEFORE you call get_info, always say a brief spoken filler out loud first, such as "Let me look that up for you." Never call the tool without speaking that acknowledgement first.

Near the start of the call, briefly remind the caller: you are not a lawyer, this is general legal information, and they should consult a licensed attorney before acting. Only answer questions about Jamaican law; politely decline anything else."""

# Spoken greeting triggered the instant the control socket opens, so the caller
# hears the bot immediately instead of dead air until they happen to speak.
GREETING_INSTRUCTIONS = (
    "Greet the caller now, out loud. Say you are JAIA, that you can answer questions "
    "about Jamaican law, that you are not a lawyer and this is general legal information "
    "so they should consult a licensed attorney before acting, then invite their question. "
    "Keep it to a few seconds and do not call any tool for this greeting."
)

GET_INFO_TOOL = {
    "type": "function",
    "name": "get_info",
    "description": (
        "Get current information relevant to the caller's question. Returns JSON "
        "with the source id (use it for citations) and the relevant Jamaican law text."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": (
                    "A DETAILED question to retrieve relevant information, "
                    "e.g. 'Fines in the 2021 Road Traffic Act'"
                ),
            },
        },
        "required": ["question"],
    },
}


def build_session_config() -> dict:
    """Realtime session configuration applied when the call is created."""
    return {
        "session": {
            "type": "realtime",
            "model": REALTIME_MODEL,
            "instructions": VOICE_INSTRUCTIONS,
            "audio": {"output": {"voice": REALTIME_VOICE}},
            "tools": [GET_INFO_TOOL],
            "tool_choice": "auto",
        }
    }


def _adjust_sdp(sdp: str) -> str:
    """Optionally coerce the DTLS setup role in the answer (see env flag)."""
    if SDP_FORCE_SETUP_ACTIVE:
        sdp = sdp.replace("a=setup:actpass", "a=setup:active")
    return sdp


# --- Azure Realtime signaling ------------------------------------------------

async def create_ephemeral_token(session: aiohttp.ClientSession) -> str:
    """Create an ephemeral Realtime session and return its client secret."""
    headers = {"api-key": REALTIME_API_KEY, "Content-Type": "application/json"}
    async with session.post(
        CLIENT_SECRETS_URL, headers=headers, json=build_session_config()
    ) as resp:
        body = await resp.text()
        if resp.status not in (200, 201):
            raise RuntimeError(f"client_secrets failed {resp.status}: {body}")
        data = json.loads(body)
    token = data.get("value") or data.get("client_secret", {}).get("value", "")
    if not token:
        raise RuntimeError(f"No ephemeral token in response: {data}")
    return token


async def relay_sdp(
    session: aiohttp.ClientSession, offer_sdp: str, ephemeral_token: str
) -> tuple[str, str]:
    """Relay WhatsApp's SDP offer to Azure Realtime; return (answer_sdp, call_id)."""
    headers = {
        "Authorization": f"Bearer {ephemeral_token}",
        "Content-Type": "application/sdp",
    }
    async with session.post(CALLS_URL, headers=headers, data=offer_sdp) as resp:
        answer_sdp = await resp.text()
        if resp.status not in (200, 201):
            raise RuntimeError(f"realtime /calls failed {resp.status}: {answer_sdp}")
        # Location: /openai/v1/realtime/calls/<call_id>
        location = resp.headers.get("Location", "")
    call_id = location.rstrip("/").split("/")[-1]
    if not call_id:
        raise RuntimeError(f"No call_id in Location header: {location!r}")
    logger.info("Realtime call created (call_id=%s).", call_id)
    return _adjust_sdp(answer_sdp), call_id


# --- WhatsApp Cloud API signaling -------------------------------------------

async def whatsapp_call_action(
    session: aiohttp.ClientSession,
    meta_call_id: str,
    action: str,
    sdp_answer: str | None = None,
) -> None:
    """Send a `pre_accept` / `accept` / `terminate` action to the Graph /calls API."""
    payload: dict = {
        "messaging_product": "whatsapp",
        "call_id": meta_call_id,
        "action": action,
    }
    if sdp_answer is not None:
        payload["session"] = {"sdp_type": "answer", "sdp": sdp_answer}
    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json",
    }
    async with session.post(GRAPH_CALLS_URL, headers=headers, json=payload) as resp:
        body = await resp.text()
        if resp.status >= 300:
            raise RuntimeError(f"WhatsApp {action} failed {resp.status}: {body}")
        logger.info("WhatsApp call action '%s' ok for %s.", action, meta_call_id)


# --- Control WebSocket: serve tool calls for the call's lifetime ------------

async def _run_tool(name: str, arguments: str) -> str:
    """Execute a tool the model requested and return its output string."""
    if name != "get_info":
        return json.dumps({"error": f"Unknown tool '{name}'"})
    try:
        args = json.loads(arguments or "{}")
    except json.JSONDecodeError:
        return json.dumps({"error": "Invalid tool arguments"})
    question = args.get("question", "")
    logger.info("Tool get_info called: %s", question)
    # relevant_info() is blocking (Pinecone); keep the event loop responsive.
    # Guard it: an exception here must be reported back to the model as tool
    # output, never allowed to bubble up and tear down the control socket.
    loop = asyncio.get_running_loop()
    try:
        return await loop.run_in_executor(None, relevant_info, question)
    except Exception:
        logger.exception("get_info tool failed for question: %s", question)
        return json.dumps(
            {"error": "The legal lookup failed just now. Ask the caller to try again."}
        )


async def run_control_ws(
    session: aiohttp.ClientSession, azure_call_id: str, meta_call_id: str
) -> None:
    """Hold the Realtime control socket open and service tool calls until the call ends."""
    ws_url = WS_URL_TEMPLATE.format(call_id=azure_call_id)
    headers = {"api-key": REALTIME_API_KEY}
    logger.info("Opening control socket for call %s.", meta_call_id)
    try:
        async with session.ws_connect(ws_url, headers=headers, heartbeat=20) as ws:
            # Re-assert instructions/tools defensively in case the ephemeral
            # session config didn't carry them.
            await ws.send_json({"type": "session.update", **build_session_config()})

            # Speak first. Without this the model stays silent until the caller
            # talks (server VAD), so the call opens with dead air and callers
            # hang up. This greeting also delivers the required disclaimer.
            await ws.send_json({
                "type": "response.create",
                "response": {"instructions": GREETING_INSTRUCTIONS},
            })

            async for msg in ws:
                if msg.type != aiohttp.WSMsgType.TEXT:
                    if msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.ERROR):
                        break
                    continue
                event = json.loads(msg.data)
                etype = event.get("type", "")

                if etype == "response.function_call_arguments.done":
                    output = await _run_tool(
                        event.get("name", ""), event.get("arguments", "")
                    )
                    await ws.send_json({
                        "type": "conversation.item.create",
                        "item": {
                            "type": "function_call_output",
                            "call_id": event.get("call_id"),
                            "output": output,
                        },
                    })
                    await ws.send_json({"type": "response.create"})
                elif etype == "error":
                    logger.warning("Realtime error on %s: %s", meta_call_id, event.get("error"))
    except asyncio.CancelledError:
        logger.info("Control socket for %s cancelled (call terminated).", meta_call_id)
        raise
    except Exception:
        logger.exception("Control socket for %s failed.", meta_call_id)
    finally:
        ACTIVE_CALLS.pop(meta_call_id, None)
        logger.info("Call %s finished.", meta_call_id)


# --- Orchestration -----------------------------------------------------------

async def handle_connect(call: dict) -> None:
    """Bridge one inbound WhatsApp call to a Realtime voice session."""
    meta_call_id = call.get("id", "")
    offer_sdp = (call.get("session") or {}).get("sdp", "")
    if not meta_call_id or not offer_sdp:
        logger.error("connect event missing call id or SDP offer: %s", call)
        return

    async with aiohttp.ClientSession(timeout=HTTP_TIMEOUT) as session:
        try:
            ephemeral = await create_ephemeral_token(session)
            answer_sdp, azure_call_id = await relay_sdp(session, offer_sdp, ephemeral)
            # pre_accept must precede accept, or WhatsApp rejects the call.
            await whatsapp_call_action(session, meta_call_id, "pre_accept", answer_sdp)
            await asyncio.sleep(0.2)  # let media begin establishing before accept
            await whatsapp_call_action(session, meta_call_id, "accept", answer_sdp)
        except Exception:
            logger.exception("Failed to accept call %s; terminating.", meta_call_id)
            try:
                await whatsapp_call_action(session, meta_call_id, "terminate")
            except Exception:
                pass
            return

        # Keep this ClientSession alive for the whole call (the control socket
        # lives on it), then serve tool calls until the call ends.
        task = asyncio.current_task()
        if task is not None:
            ACTIVE_CALLS[meta_call_id] = task
        await run_control_ws(session, azure_call_id, meta_call_id)


async def handle_terminate(call: dict) -> None:
    """Cancel the control socket for a call WhatsApp reports as ended."""
    meta_call_id = call.get("id", "")
    task = ACTIVE_CALLS.get(meta_call_id)
    if task:
        task.cancel()
        logger.info("Terminating call %s on WhatsApp's request.", meta_call_id)


# --- HTTP surface (called by the Functions webhook) -------------------------

async def start_call(request: web.Request) -> web.Response:
    """Entry point invoked by the WhatsApp webhook for each `calls` event."""
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "invalid json"}, status=400)

    call = body.get("call", body)
    event = call.get("event") or call.get("status")

    if event == "connect":
        # Run the bridge as a background task so we ack the webhook immediately.
        _spawn(handle_connect(call))
    elif event == "terminate":
        _spawn(handle_terminate(call))
    else:
        logger.info("Ignoring call event '%s' for %s.", event, call.get("id"))

    return web.json_response({"status": "accepted"})


async def health(_: web.Request) -> web.Response:
    return web.json_response({"status": "ok", "active_calls": len(ACTIVE_CALLS)})


def create_app() -> web.Application:
    app = web.Application()
    app.router.add_post("/start_call", start_call)
    app.router.add_get("/health", health)
    return app


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    logger.info("Starting calling bridge on :%d (model=%s).", port, REALTIME_MODEL)
    web.run_app(create_app(), host="0.0.0.0", port=port)
