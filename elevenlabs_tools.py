"""Webhook tool endpoint for the ElevenLabs voice agent.

ElevenLabs Agents answers WhatsApp voice calls natively (the WABA is imported
into the ElevenLabs platform and an agent is assigned to the number). To ground
its answers in Jamaican law, the agent is configured with a webhook tool named
`get_info` that POSTs here — the same `relevant_info()` lookup the chat bot and
the old gpt-realtime bridge use.

Requests are authenticated with a static bearer secret: configure the tool in
the ElevenLabs dashboard with Bearer Token auth using `ELEVENLABS_TOOL_SECRET`.
"""

import hmac
import json
import logging
import os

import azure.functions as func
from dotenv import load_dotenv

from relevant_info import relevant_info

load_dotenv()

# Shared secret the ElevenLabs tool sends as `Authorization: Bearer <secret>`.
ELEVENLABS_TOOL_SECRET = os.getenv("ELEVENLABS_TOOL_SECRET", "")

bp = func.Blueprint()


def _authorized(req: func.HttpRequest) -> bool:
    if not ELEVENLABS_TOOL_SECRET:
        logging.error("ELEVENLABS_TOOL_SECRET is not set; rejecting get_info request.")
        return False
    supplied = req.headers.get("Authorization", "")
    expected = f"Bearer {ELEVENLABS_TOOL_SECRET}"
    return hmac.compare_digest(supplied, expected)


@bp.route(route="get_info", methods=[func.HttpMethod.POST])
def get_info(req: func.HttpRequest) -> func.HttpResponse:
    """Serve a `get_info` tool call from the ElevenLabs agent.

    Expects `{"question": "..."}` and returns the relevant Jamaican law as JSON
    (id + text per source). Lookup failures return 200 with an `error` field so
    the agent can tell the caller to try again instead of hearing dead air.
    """
    if not _authorized(req):
        return func.HttpResponse("Unauthorized", status_code=401)

    try:
        body = req.get_json()
    except ValueError:
        return func.HttpResponse(
            json.dumps({"error": "Invalid JSON body"}),
            status_code=400,
            mimetype="application/json",
        )

    question = (body or {}).get("question", "").strip()
    if not question:
        return func.HttpResponse(
            json.dumps({"error": "Missing 'question' field"}),
            status_code=400,
            mimetype="application/json",
        )

    logging.info(f"ElevenLabs get_info tool called: {question}")
    try:
        law_json = relevant_info(question)
    except Exception:
        logging.exception(f"get_info lookup failed for question: {question}")
        return func.HttpResponse(
            json.dumps(
                {"error": "The legal lookup failed just now. Ask the caller to try again."}
            ),
            status_code=200,
            mimetype="application/json",
        )

    return func.HttpResponse(law_json, status_code=200, mimetype="application/json")
