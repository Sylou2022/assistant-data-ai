"""Serveur : connexion Google obligatoire + notification à chaque connexion.

Lancement : python -m app.server
"""
from __future__ import annotations

import logging
import os
import threading

import gradio as gr
import httpx
import uvicorn
from authlib.integrations.starlette_client import OAuth
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from starlette.middleware.sessions import SessionMiddleware

from app.App_IA_Assistant_Gradio import (
    CSS, FORCE_LIGHT_JS, HAS_LOGO, IMAGE_DIR, LOGO_PATH, THEME, build_app, get_pipeline,
)

logger = logging.getLogger(__name__)

# Variables d'environnement
GOOGLE_CLIENT_ID = os.environ["GOOGLE_CLIENT_ID"]
GOOGLE_CLIENT_SECRET = os.environ["GOOGLE_CLIENT_SECRET"]
SESSION_SECRET = os.environ["SESSION_SECRET"]          # chaîne aléatoire longue
NTFY_TOPIC = os.environ.get("NTFY_TOPIC", "")          # ex. assistant-data-ai-x7k2q9 (secret)
ALLOWED_EMAILS = {
    e.strip().lower() for e in os.environ.get("ALLOWED_EMAILS", "").split(",") if e.strip()
}                                                       # vide = tout compte Gmail accepté

oauth = OAuth()
oauth.register(
    name="google",
    client_id=GOOGLE_CLIENT_ID,
    client_secret=GOOGLE_CLIENT_SECRET,
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={"scope": "openid email profile"},
)

app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key=SESSION_SECRET, https_only=True, same_site="lax")


# ---------------- Notification ----------------

NTFY_TOKEN = os.environ.get("tk_7is0n0tiyhxrifum61w0vrh61la02", "")

def notify(title: str, message: str) -> None:
    if not NTFY_TOPIC:
        logger.warning("NTFY_TOPIC vide : aucune notification envoyée.")
        return

    def _send():
        headers = {"Title": title.encode("utf-8"), "Tags": "bell"}
        if NTFY_TOKEN:
            headers["Authorization"] = f"Bearer {NTFY_TOKEN}"
        try:
            r = httpx.post(
                f"https://ntfy.sh/{NTFY_TOPIC}",
                content=message.encode("utf-8"),
                headers=headers,
                timeout=10,
            )
            logger.warning("ntfy : HTTP %s", r.status_code)
        except Exception:
            logger.exception("Notification impossible")

    threading.Thread(target=_send, daemon=True).start()


# ---------------- Authentification ----------------
def get_user(request: Request) -> str | None:
    user = request.session.get("user")
    return user["email"] if user else None


@app.get("/")
def home(request: Request):
    return RedirectResponse("/app" if get_user(request) else "/login")


from fastapi.responses import FileResponse, Response


@app.get("/generated-image.png", include_in_schema=False)
def logo():
    if HAS_LOGO:
        return FileResponse(LOGO_PATH)
    return Response(status_code=404)


@app.get("/login", response_class=HTMLResponse)
def login_page():
    logo_html = (
        '<img src="/generated-image.png" alt="Assistant Data AI" '
        'style="width:84px;height:84px;object-fit:contain;margin-bottom:14px">'
        if HAS_LOGO else ""
    )
    return f"""
    <div style="font-family:system-ui;max-width:380px;margin:12vh auto;text-align:center">
      {logo_html}
      <h2>Assistant Data AI</h2>
      <p>Connexion requise avec un compte Google.</p>
      <a href="/login/google"
         style="display:inline-block;padding:12px 22px;background:#4f46e5;color:#fff;
                border-radius:10px;text-decoration:none;font-weight:600">
        Se connecter avec Google
      </a>
    </div>"""


@app.get("/login/google")
async def login_google(request: Request):
    redirect_uri = str(request.url_for("auth"))
    return await oauth.google.authorize_redirect(request, redirect_uri)


@app.get("/auth", name="auth")
async def auth(request: Request):
    token = await oauth.google.authorize_access_token(request)
    info = token.get("userinfo") or {}
    email = (info.get("email") or "").lower()

    if not email or not info.get("email_verified"):
        return HTMLResponse("Compte Google non vérifié.", status_code=403)
    if ALLOWED_EMAILS and email not in ALLOWED_EMAILS:
        notify("⛔ Accès refusé", f"{email} a tenté de se connecter.")
        return HTMLResponse("Accès non autorisé pour ce compte.", status_code=403)

    request.session["user"] = {"email": email, "name": info.get("name", "")}
    ip = request.headers.get("x-forwarded-for", request.client.host if request.client else "?")
    notify("🔔 Assistant Data AI consulté", f"{info.get('name', email)} ({email})\nIP : {ip}")
    return RedirectResponse("/app")


@app.get("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/login")


@app.exception_handler(401)
async def unauthorized(request: Request, exc):
    return RedirectResponse("/login")



from fastapi.responses import FileResponse

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return FileResponse(LOGO_PATH)

# ---------------- Gradio monté derrière l'authentification ----------------
app = gr.mount_gradio_app(
    app,
    build_app(),
    path="/app",
    auth_dependency=get_user,
    theme=THEME,
    css=CSS,
    js=FORCE_LIGHT_JS,
    # favicon_path=str(LOGO_PATH) if HAS_LOGO else None,
    allowed_paths=[str(IMAGE_DIR)],
)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        get_pipeline()
    except Exception:
        logger.exception("Préchauffage du pipeline impossible")
    uvicorn.run(
        app, host="0.0.0.0", port=int(os.environ.get("PORT", 7860)),
        proxy_headers=True, forwarded_allow_ips="*",
    )