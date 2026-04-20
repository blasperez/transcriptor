import os
import uuid
import tempfile
import subprocess
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from groq import Groq

BASE_DIR = Path(__file__).parent

app = FastAPI(title="Video Transcriptor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Almacenamiento en memoria (se reinicia si Render hace restart)
transcriptions: dict[str, dict] = {}


class TranscribeRequest(BaseModel):
    url: str
    language: str = "es"


@app.api_route("/", methods=["GET", "HEAD"], response_class=HTMLResponse)
def index():
    return HTMLResponse((BASE_DIR / "index.html").read_text(encoding="utf-8"))


@app.get("/ping")
def ping():
    return {"status": "ok"}


@app.post("/transcribe")
def transcribe(req: TranscribeRequest):
    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = os.path.join(tmpdir, "audio.mp3")

        result = subprocess.run(
            [
                "yt-dlp",
                "--no-playlist",
                "--extract-audio",
                "--audio-format", "mp3",
                "--audio-quality", "5",
                "--output", audio_path,
                "--max-filesize", "25m",
                req.url,
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )

        if result.returncode != 0:
            raise HTTPException(
                status_code=400,
                detail=f"No se pudo descargar el video: {result.stderr[-300:]}"
            )

        actual_path = audio_path
        if not os.path.exists(actual_path):
            matches = [f for f in os.listdir(tmpdir) if "audio" in f]
            if not matches:
                raise HTTPException(status_code=400, detail="Audio no encontrado tras la descarga")
            actual_path = os.path.join(tmpdir, matches[0])

        with open(actual_path, "rb") as f:
            transcript = client.audio.transcriptions.create(
                model="whisper-large-v3-turbo",
                file=f,
                language=req.language if req.language != "auto" else None,
                response_format="text",
            )

    tid = str(uuid.uuid4())[:8]
    # Groq con response_format="text" devuelve string directo
    text = transcript if isinstance(transcript, str) else transcript.text
    transcriptions[tid] = {"url": req.url, "text": text}

    return {"id": tid, "transcript": text}


@app.get("/go", response_class=HTMLResponse)
def transcribe_direct(url: str, language: str = "es"):
    req = TranscribeRequest(url=url, language=language)
    data = transcribe(req)
    tid = data["id"]
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url=f"/t/{tid}", status_code=302)


@app.get("/t/{tid}", response_class=HTMLResponse)
def view_transcription(tid: str):
    entry = transcriptions.get(tid)
    if not entry:
        return HTMLResponse("""<!DOCTYPE html><html lang="es"><head><meta charset="UTF-8">
<title>Expirada</title><style>body{font-family:Arial,sans-serif;max-width:600px;margin:60px auto;padding:0 20px;text-align:center}</style></head>
<body><h2>⏱️ Transcripción expirada</h2>
<p>El servidor se reinició y la transcripción se perdió.<br>Vuelve a transcribir el video.</p>
<a href="/">← Volver al inicio</a></body></html>""", status_code=410)

    text_escaped = entry["text"].replace("<", "&lt;").replace(">", "&gt;")
    url_escaped = entry["url"].replace("<", "&lt;").replace(">", "&gt;")

    return HTMLResponse(f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Transcripción</title>
<style>
  body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 40px auto; padding: 0 20px; line-height: 1.7; color: #222; }}
  .source {{ font-size: 12px; color: #888; margin-bottom: 16px; word-break: break-all; }}
  .transcript {{ white-space: pre-wrap; font-size: 15px; }}
</style>
</head>
<body>
<div class="source">Fuente: {url_escaped}</div>
<div class="transcript">{text_escaped}</div>
</body>
</html>""")
