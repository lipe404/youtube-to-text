import os
import re
import tempfile
import uuid
from contextlib import contextmanager
from typing import Tuple

from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from yt_dlp import YoutubeDL
from faster_whisper import WhisperModel
import subprocess
import imageio_ffmpeg


ALLOWED_HOSTS = {
    "youtube.com",
    "www.youtube.com",
    "youtu.be",
    "m.youtube.com",
}


def create_app() -> Flask:
    app = Flask(__name__)

    app.config["SECRET_KEY"] = os.environ.get(
        "FLASK_SECRET_KEY", "dev-secret-change-me"
    )
    app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB

    @app.route("/", methods=["GET"])
    def index():
        return render_template("index.html", transcript=None)

    @app.route("/transcribe", methods=["POST"])
    def transcribe():
        input_url = (request.form.get("youtube_url") or "").strip()

        try:
            validate_youtube_url_or_raise(input_url)
        except ValueError as exc:
            flash(str(exc), "error")
            return redirect(url_for("index"))

        # Download vídeo como MP4
        try:
            file_path, filename = download_youtube_video(input_url)
        except Exception as exc:
            flash(f"Falha ao baixar o vídeo: {exc}", "error")
            return redirect(url_for("index"))

        # Transcrever com Whisper
        try:
            transcript_text = transcribe_with_whisper_local(file_path)
        except Exception as exc:
            flash(f"Falha na transcrição: {exc}", "error")
            transcript_text = None
        finally:
            try:
                os.remove(file_path)
            except OSError:
                pass

        return render_template("index.html", transcript=transcript_text)

    return app


def validate_youtube_url_or_raise(url: str) -> None:
    """Valida se a URL fornecida é do YouTube."""
    if not url:
        raise ValueError("Informe uma URL do YouTube.")

    # Padrão para URLs do YouTube
    pattern = re.compile(r"^https?://([^/]+)(/.*)?$", re.IGNORECASE)
    match = pattern.match(url)

    if not match:
        raise ValueError("URL inválida.")

    host = match.group(1).lower()
    if host not in ALLOWED_HOSTS:
        raise ValueError("A URL deve ser do domínio youtube.com ou youtu.be.")

    # Verifica se contém ID de vídeo
    video_id = extract_video_id(url)
    if not video_id:
        raise ValueError("Forneça uma URL válida de vídeo do YouTube.")


def extract_video_id(url: str) -> str:
    """Extrai o ID do vídeo da URL do YouTube."""
    patterns = [
        r"youtu\.be/([A-Za-z0-9_-]{11})",          # youtu.be/
        r"embed/([A-Za-z0-9_-]{11})",              # embed/
        r"shorts/([A-Za-z0-9_-]{11})",             # shorts/
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


@contextmanager
def temp_download_dir():
    """Cria diretório temporário para downloads."""
    base_tmp = tempfile.gettempdir()
    session_dir = os.path.join(base_tmp, f"youtube_to_text_{uuid.uuid4().hex}")
    os.makedirs(session_dir, exist_ok=True)
    try:
        yield session_dir, base_tmp
    finally:
        try:
            os.rmdir(session_dir)
        except OSError:
            pass


def download_youtube_video(url: str) -> Tuple[str, str]:
    """Baixa vídeo do YouTube como MP4 usando yt-dlp.

    Returns:
        Tuple[str, str]: (caminho_do_arquivo, nome_do_arquivo)
    """
    with temp_download_dir() as (session_dir, _):
        # Obter caminho do ffmpeg empacotado
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        ffmpeg_dir = os.path.dirname(ffmpeg_exe)

        unique = uuid.uuid4().hex
        outtmpl = os.path.join(session_dir, f"%(title).80s-{unique}.%(ext)s")

        ydl_opts = {
            "outtmpl": outtmpl,
            # Preferir formatos já em MP4 sem necessidade de merge
            "format": "best[ext=mp4]/bestvideo[ext=mp4]+bestaudio[ext=m4a]/best",
            "restrictfilenames": True,
            "nocheckcertificate": True,
            "quiet": True,
            "no_warnings": True,
            "retries": 3,
            "fragment_retries": 3,
            "ratelimit": 10_000_000,  # ~10MB/s
            "trim_filenames": 120,
            "max_filesize": 500 * 1024 * 1024,  # 500 MB
            "merge_output_format": "mp4",
            # CRÍTICO: Configurar ffmpeg empacotado
            "ffmpeg_location": ffmpeg_dir,
            # Evitar abort em erro de merge (fallback para best single)
            "ignoreerrors": False,
            "abort_on_error": False,
        }

        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)

            ext = info.get("ext") or "mp4"
            title = info.get("title") or f"youtube-{unique}"
            safe_name = secure_filename(f"{title}-{unique}.{ext}")

            # Encontra o arquivo baixado
            candidates = [
                os.path.join(session_dir, f)
                for f in os.listdir(session_dir)
                if f.endswith((".mp4", ".webm", ".mkv", f".{ext}"))
            ]

            if not candidates:
                raise RuntimeError(
                    "Não foi possível localizar o arquivo baixado.")

            # Pega o maior arquivo (mais provável ser o vídeo completo)
            candidates.sort(key=lambda p: os.path.getsize(p), reverse=True)
            file_path = candidates[0]

            return file_path, safe_name


def extract_wav_with_bundled_ffmpeg(input_video_path: str) -> str:
    """Extrai áudio como WAV 16kHz mono usando ffmpeg empacotado."""
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    wav_path = os.path.splitext(input_video_path)[0] + "_audio.wav"

    cmd = [
        ffmpeg_exe,
        "-y",
        "-i", input_video_path,
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-f", "wav",
        wav_path,
    ]

    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=300  # 5 minutos timeout
    )

    if proc.returncode != 0 or not os.path.exists(wav_path):
        stderr = proc.stderr.decode('utf-8', errors='ignore')
        raise RuntimeError(
            f"Falha ao extrair áudio com ffmpeg: {stderr[:200]}")

    return wav_path


def transcribe_with_whisper_local(file_path: str) -> str:
    """Transcreve áudio usando Whisper local (faster-whisper)."""
    wav_path = extract_wav_with_bundled_ffmpeg(file_path)

    try:
        # Usa modelo configurável via env var
        model_size = os.environ.get("WHISPER_MODEL", "base")
        model = WhisperModel(
            model_size,
            device="cpu",
            compute_type="int8"
        )

        # Transcrição com detecção automática de idioma
        segments, info = model.transcribe(
            wav_path,
            beam_size=3,  # Melhor qualidade que beam_size=1
            language=None,  # Detecção automática
            vad_filter=True,  # Remove silêncios
            vad_parameters=dict(
                min_silence_duration_ms=500
            )
        )

        # Monta texto com timestamps opcionais
        text_parts = []
        for seg in segments:
            text_parts.append(seg.text)

        transcript = " ".join(part.strip()
                              for part in text_parts if part.strip())

        if not transcript:
            raise RuntimeError("Nenhuma fala detectada no vídeo.")

        return transcript

    finally:
        try:
            os.remove(wav_path)
        except OSError:
            pass


if __name__ == "__main__":
    app = create_app()
    port = int(os.environ.get("PORT", "8000"))
    app.run(host="0.0.0.0", port=port, debug=True)
