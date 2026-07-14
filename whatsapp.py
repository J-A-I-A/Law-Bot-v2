import os
import json
import logging
import requests
import azure.functions as func
from dotenv import load_dotenv

import bot
import messages

load_dotenv()

# WhatsApp Cloud API credentials
WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN")
WHATSAPP_PHONE_ID = os.getenv("WHATAPP_PHONE_ID")  # NOTE: env var name as spelled in .env
# Token you set in the Meta App webhook configuration (GET verification handshake)
VERIFY_TOKEN = os.getenv("WHATSAPP_VERIFY_TOKEN")

# Always-on calling bridge (see calling.py). Meta delivers both `messages` and
# `calls` events to this single webhook URL, so voice calls are forwarded here.
CALLING_BRIDGE_URL = os.getenv("CALLING_BRIDGE_URL")

GRAPH_API_VERSION = "v25.0"
GRAPH_URL = f"https://graph.facebook.com/{GRAPH_API_VERSION}/{WHATSAPP_PHONE_ID}/messages"

# WhatsApp hard limit for a text message body
MESSAGE_LIMIT = 4096

# Storage queue that decouples the (fast) webhook from the (slow) bot processing.
# `connection` refers to an app setting; AzureWebJobsStorage is always present in a
# Functions app. Swap it for AZURE_STORAGE_CONNECTION_STRING if you prefer.
QUEUE_NAME = "whatsapp-messages"
QUEUE_CONNECTION = "AzureWebJobsStorage"

bp = func.Blueprint()


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json",
    }


def send_typing_indicator(message_id: str) -> None:
    """Mark the incoming message as read and show the typing indicator.

    The indicator is displayed for up to 25 seconds, or until a message is
    sent to the user (whichever comes first), so it should be triggered right
    before the bot starts generating its answer.
    """
    payload = {
        "messaging_product": "whatsapp",
        "status": "read",
        "message_id": message_id,
        "typing_indicator": {"type": "text"},
    }
    try:
        response = requests.post(GRAPH_URL, headers=_headers(), json=payload, timeout=15)
        response.raise_for_status()
    except requests.RequestException as exc:
        logging.warning(f"Failed to send typing indicator: {exc}")


def send_message(to: str, body: str) -> None:
    """Send a text message to the user, splitting it if it exceeds the limit."""
    for chunk in split_message(body):
        payload = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": to,
            "type": "text",
            "text": {"preview_url": False, "body": chunk},
        }
        try:
            response = requests.post(GRAPH_URL, headers=_headers(), json=payload, timeout=30)
            response.raise_for_status()
            logging.info(f"Message sent to {to}.")
        except requests.RequestException as exc:
            logging.error(f"Failed to send message to {to}: {exc}")


def split_message(text: str, limit: int = MESSAGE_LIMIT) -> list:
    """Split text into chunks no longer than `limit`, preferring newline breaks."""
    if not text:
        return [""]

    chunks = []
    remaining = text
    while len(remaining) > limit:
        window = remaining[:limit]
        split_at = window.rfind("\n")
        if split_at == -1:
            split_at = limit
        chunks.append(remaining[:split_at])
        remaining = remaining[split_at:].lstrip("\n")
    chunks.append(remaining)
    return chunks


def _extract_message(payload: dict):
    """Pull the first incoming message out of a webhook payload, if present.

    Returns the message dict, or None for status callbacks / non-message events.
    """
    try:
        value = payload["entry"][0]["changes"][0]["value"]
    except (KeyError, IndexError, TypeError):
        return None
    messages_list = value.get("messages")
    if not messages_list:
        return None
    return messages_list[0]


def _extract_call(payload: dict):
    """Pull the first inbound call event out of a webhook payload, if present.

    WhatsApp calling events arrive under the `calls` field with the same webhook
    envelope as messages. Returns the call dict, or None for non-call events.
    """
    try:
        value = payload["entry"][0]["changes"][0]["value"]
    except (KeyError, IndexError, TypeError):
        return None
    calls_list = value.get("calls")
    if not calls_list:
        return None
    return calls_list[0]


def _forward_call_to_bridge(call: dict) -> None:
    """Hand a call event to the always-on calling bridge and return immediately."""
    if not CALLING_BRIDGE_URL:
        logging.error("Received a call event but CALLING_BRIDGE_URL is not set; ignoring.")
        return
    try:
        response = requests.post(
            f"{CALLING_BRIDGE_URL.rstrip('/')}/start_call",
            json={"call": call},
            timeout=10,
        )
        response.raise_for_status()
        logging.info(f"Forwarded call {call.get('id')} ({call.get('event')}) to the bridge.")
    except requests.RequestException as exc:
        logging.error(f"Failed to forward call to bridge: {exc}")


@bp.route(
    route="whatsapp_webhook",
    methods=[func.HttpMethod.GET, func.HttpMethod.POST],
)
@bp.queue_output(arg_name="outqueue", queue_name=QUEUE_NAME, connection=QUEUE_CONNECTION)
def whatsapp_webhook(req: func.HttpRequest, outqueue: func.Out[str]) -> func.HttpResponse:
    """Fast path: verify, acknowledge, and hand off to the queue.

    Meta retries any webhook that does not return 200 within a few seconds, so this
    function never calls the model. It only validates the request, shows the typing
    indicator, enqueues the work, and returns 200 immediately.
    """
    # --- Webhook verification handshake (Meta sends a GET on setup) ---
    if req.method == "GET":
        mode = req.params.get("hub.mode")
        token = req.params.get("hub.verify_token")
        challenge = req.params.get("hub.challenge")
        if mode == "subscribe" and token == VERIFY_TOKEN:
            logging.info("WhatsApp webhook verified.")
            return func.HttpResponse(challenge, status_code=200)
        logging.warning("WhatsApp webhook verification failed: invalid mode or token.")
        return func.HttpResponse("Verification failed", status_code=403)

    # --- Incoming message handling ---
    try:
        payload = req.get_json()
    except ValueError:
        logging.warning("Received a webhook POST with an invalid JSON body.")
        return func.HttpResponse("Bad Request", status_code=400)

    # Voice calls arrive under the `calls` field. Forward them to the always-on
    # bridge (which does the WebRTC/Realtime work) and acknowledge immediately.
    call = _extract_call(payload)
    if call is not None:
        logging.info(f"Received WhatsApp call event '{call.get('event')}' ({call.get('id')}).")
        _forward_call_to_bridge(call)
        return func.HttpResponse(status_code=200)

    message = _extract_message(payload)
    # Status callbacks (sent/delivered/read) and other events: acknowledge and ignore.
    if message is None:
        logging.info("Webhook event contained no user message (status callback); ignoring.")
        return func.HttpResponse(status_code=200)

    from_number = message.get("from")
    message_id = message.get("id")
    logging.info(f"Received WhatsApp message from {from_number}.")

    if message.get("type") != "text":
        logging.info(f"Non-text message from {from_number} (type: {message.get('type')}); replying with a prompt.")
        send_message(from_number, "I currently only understand text messages, please only send text messages.")
        return func.HttpResponse(status_code=200)

    question = message["text"]["body"]

    # Show the typing indicator immediately (also marks the message as read) so the
    # user gets instant feedback while the queued worker generates the answer.
    if message_id:
        send_typing_indicator(message_id)

    outqueue.set(json.dumps({"from": from_number, "question": question}))
    logging.info(f"Queued question from {from_number}.")

    return func.HttpResponse(status_code=200)


@bp.queue_trigger(arg_name="msg", queue_name=QUEUE_NAME, connection=QUEUE_CONNECTION)
def process_message(msg: func.QueueMessage) -> None:
    """Slow path: run the bot and deliver the reply, off the webhook request thread."""
    data = json.loads(msg.get_body().decode("utf-8"))
    from_number = data["from"]
    question = data["question"]

    logging.info(f"Worker picked up queued question from {from_number}.")
    past_conversations = messages.get_messages(from_number)
    response = bot.Law_bot(previous_message=past_conversations, question=question)
    logging.info(f"Bot processed question from {from_number}.")

    if response:
        messages.add_messages(phone_number=from_number, question=question, response=response)
        send_message(from_number, response)
    else:
        send_message(from_number, "Currently Offline Please Try Again Later")
