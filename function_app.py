import azure.functions as func

from whatsapp import bp
from elevenlabs_tools import bp as elevenlabs_bp

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)
# The WhatsApp webhook route sets its own ANONYMOUS auth level so Meta can reach it;
# requests are gated by the verify-token handshake instead. The ElevenLabs
# get_info tool route is gated by a bearer secret (ELEVENLABS_TOOL_SECRET).
app.register_functions(bp)
app.register_functions(elevenlabs_bp)