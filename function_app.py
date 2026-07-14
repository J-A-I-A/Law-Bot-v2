import azure.functions as func

from whatsapp import bp

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)
# The WhatsApp webhook route sets its own ANONYMOUS auth level so Meta can reach it;
# requests are gated by the verify-token handshake instead.
app.register_functions(bp)