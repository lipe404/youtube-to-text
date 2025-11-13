import os
import re
import tempfile
import uuid
import shutil
from contextlib import contextmanager
from typing import Tuple, Optional

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

# Limites de recursos
MAX_VIDEO_DURATION = 600  # 10 minutos
MAX_VIDEO_SIZE_MB = 100  # 100 MB


def create_app() -> Flask:
    app = Flask(__name__)

    app.config["SECRET_KEY"] = os.environ.get(
        "FLASK_SECRET_KEY", "dev-secret-change-me"
    )
    app.config["MAX_CONTENT_LENGTH"] = MAX_VIDEO_SIZE_MB * 1024 * 1024

    @app.route("/", methods=["GET"])
    def index():
        return render_template("index.html", transcript=None)

    @app.route("/transcribe", methods=["POST"])
    def transcribe():
        input_url = (request.form.get("youtube_url") or "").strip()

        if not input_url:
            flash("Informe uma URL do YouTube.", "error")
            return redirect(url_for("index"))

        try:
            validate_youtube_url_or_raise(input_url)
        except ValueError as exc:
            flash(str(exc), "error")
            return redirect(url_for("index"))

        file_path = None
        wav_path = None

        try:
            # Download apenas áudio
            file_path = download_youtube_audio_optimized(input_url)

            # Validar duração
            duration = get_audio_duration(file_path)
            if duration > MAX_VIDEO_DURATION:
                raise ValueError(
                    f"Vídeo muito longo ({int(duration)}s). Máximo: {MAX_VIDEO_DURATION}s"
                )

            # Transcrever
            transcript_text = transcribe_with_whisper_local(file_path)

            flash("Transcrição concluída com sucesso!", "success")
            return render_template("index.html", transcript=transcript_text)

        except Exception as exc:
            error_msg = str(exc)

            # Mensagens amigáveis para erros comuns
            if "Sign in to confirm" in error_msg or "bot" in error_msg.lower():
                error_msg = (
                    "YouTube bloqueou a requisição (detecção de bot). "
                    "Soluções: (1) Aguarde 10-15 minutos e tente novamente, "
                    "(2) Use outra URL, (3) Configure cookies (veja documentação)."
                )
            elif "Video unavailable" in error_msg:
                error_msg = "Vídeo indisponível, privado ou restrito por região."
            elif "Requested format not available" in error_msg:
                error_msg = "Formato de vídeo não disponível para download."

            flash(f"Erro: {error_msg}", "error")
            return redirect(url_for("index"))

        finally:
            cleanup_files([file_path, wav_path])

    @app.route("/upload", methods=["POST"])
    def upload():
        """Rota alternativa: upload direto de arquivo."""
        if "video_file" not in request.files:
            flash("Nenhum arquivo enviado.", "error")
            return redirect(url_for("index"))

        file = request.files["video_file"]

        if file.filename == "":
            flash("Arquivo inválido.", "error")
            return redirect(url_for("index"))

        # Validar extensão
        allowed_extensions = {'.mp4', '.mp3', '.wav', '.m4a', '.webm', '.ogg'}
        file_ext = os.path.splitext(file.filename)[1].lower()

        if file_ext not in allowed_extensions:
            flash(
                f"Formato não suportado. Use: {', '.join(allowed_extensions)}",
                "error"
            )
            return redirect(url_for("index"))

        temp_path = None

        try:
            # Salvar temporariamente
            temp_path = os.path.join(
                tempfile.gettempdir(),
                f"upload_{uuid.uuid4().hex}{file_ext}"
            )
            file.save(temp_path)

            # Validar tamanho
            file_size_mb = os.path.getsize(temp_path) / (1024 * 1024)
            if file_size_mb > MAX_VIDEO_SIZE_MB:
                raise ValueError(
                    f"Arquivo muito grande ({file_size_mb:.1f}MB). "
                    f"Máximo: {MAX_VIDEO_SIZE_MB}MB"
                )

            # Transcrever
            transcript = transcribe_with_whisper_local(temp_path)

            flash("Transcrição concluída com sucesso!", "success")
            return render_template("index.html", transcript=transcript)

        except Exception as exc:
            flash(f"Erro: {exc}", "error")
            return redirect(url_for("index"))

        finally:
            cleanup_files([temp_path])

    return app


def validate_youtube_url_or_raise(url: str) -> None:
    """Valida se a URL fornecida é do YouTube."""
    if not url:
        raise ValueError("Informe uma URL do YouTube.")

    pattern = re.compile(r"^https?://([^/]+)(/.*)?$", re.IGNORECASE)
    match = pattern.match(url)

    if not match:
        raise ValueError("URL inválida.")

    host = match.group(1).lower()
    if host not in ALLOWED_HOSTS:
        raise ValueError("A URL deve ser do domínio youtube.com ou youtu.be.")

    video_id = extract_video_id(url)
    if not video_id:
        raise ValueError("Forneça uma URL válida de vídeo do YouTube.")


def extract_video_id(url: str) -> Optional[str]:
    """Extrai o ID do vídeo da URL do YouTube."""
    patterns = [
        r"(?:v=|/)([0-9A-Za-z_-]{11}).*",  # Padrão geral
        r"youtu\.be/([0-9A-Za-z_-]{11})",   # youtu.be
        r"embed/([0-9A-Za-z_-]{11})",       # embed
        r"shorts/([0-9A-Za-z_-]{11})",      # shorts
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def download_youtube_audio_optimized(url: str) -> str:
    """
    Download otimizado de áudio do YouTube com estratégias anti-bot.
    """
    temp_dir = tempfile.mkdtemp(prefix="yt_audio_")
    unique = uuid.uuid4().hex[:8]

    output_path = os.path.join(temp_dir, f"audio_{unique}.%(ext)s")

    # Configurações anti-detecção de bot
    ydl_opts = {
        "outtmpl": output_path,
        "format": "bestaudio[ext=m4a]/bestaudio/best",  # Apenas áudio

        # === ESTRATÉGIAS ANTI-BOT ===

        # 1. User-Agent moderno e realista
        "http_headers": {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/131.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Cache-Control": "max-age=0",
        },

        # 2. Cookies (se disponível)
        "cookiefile": os.environ.get("YOUTUBE_COOKIES_PATH"),

        # 3. Extractor args para YouTube
        "extractor_args": {
            "youtube": {
                "player_client": ["android", "web"],  # Múltiplos clients
                # Pular verificações pesadas
                "player_skip": ["webpage", "configs"],
            }
        },

        # 4. Configurações de rede
        "socket_timeout": 30,
        "retries": 3,
        "fragment_retries": 3,
        "file_access_retries": 3,
        "ratelimit": 5_000_000,  # 5MB/s

        # 5. Configurações gerais
        "quiet": True,
        "no_warnings": True,
        "ignoreerrors": False,
        "nocheckcertificate": True,

        # 6. Limites
        "max_filesize": MAX_VIDEO_SIZE_MB * 1024 * 1024,

        # 7. Pós-processamento mínimo
        "postprocessors": [],
        "extract_flat": False,

        # 8. Geo bypass (tenta contornar restrições regionais)
        "geo_bypass": True,
        "geo_bypass_country": "US",
    }

    try:
        with YoutubeDL(ydl_opts) as ydl:
            # Extrair info primeiro (mais leve)
            info = ydl.extract_info(url, download=False)

            # Verificar duração antes de baixar
            duration = info.get("duration", 0)
            if duration > MAX_VIDEO_DURATION:
                raise ValueError(
                    f"Vídeo muito longo ({duration}s). Máximo: {MAX_VIDEO_DURATION}s"
                )

            # Agora sim, baixar
            ydl.download([url])

            # Localizar arquivo baixado
            downloaded_file = find_downloaded_file(temp_dir)
            if not downloaded_file:
                raise RuntimeError("Arquivo não foi baixado corretamente.")

            return downloaded_file

    except Exception as e:
        cleanup_files([temp_dir])
        raise


def find_downloaded_file(directory: str) -> Optional[str]:
    """Localiza o arquivo de áudio baixado."""
    audio_extensions = ('.m4a', '.mp3', '.opus', '.webm', '.ogg', '.wav')

    for filename in os.listdir(directory):
        if filename.endswith(audio_extensions):
            return os.path.join(directory, filename)

    return None


def get_audio_duration(file_path: str) -> float:
    """Obtém duração do áudio em segundos usando ffprobe."""
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    ffprobe = ffmpeg_exe.replace('ffmpeg', 'ffprobe')

    cmd = [
        ffprobe,
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        file_path
    ]

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10
        )
        duration_str = result.stdout.strip()
        return float(duration_str) if duration_str else 0.0
    except Exception:
        return 0.0


def extract_wav_with_bundled_ffmpeg(input_file_path: str) -> str:
    """Extrai áudio como WAV 16kHz mono usando ffmpeg empacotado."""
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    wav_path = os.path.splitext(input_file_path)[0] + "_audio.wav"

    cmd = [
        ffmpeg_exe,
        "-y",
        "-i", input_file_path,
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
        timeout=300  # 5 minutos
    )

    if proc.returncode != 0 or not os.path.exists(wav_path):
        stderr = proc.stderr.decode('utf-8', errors='ignore')
        raise RuntimeError(
            f"Falha ao extrair áudio: {stderr[:200]}"
        )

    return wav_path


def transcribe_with_whisper_local(file_path: str) -> str:
    """Transcreve áudio usando Whisper otimizado."""
    wav_path = None

    try:
        # Extrair WAV
        wav_path = extract_wav_with_bundled_ffmpeg(file_path)

        # Otimização para produção
        is_production = os.environ.get("FLASK_ENV") == "production"
        model_size = "tiny" if is_production else os.environ.get(
            "WHISPER_MODEL", "base")

        # Carregar modelo
        model = WhisperModel(
            model_size,
            device="cpu",
            compute_type="int8",
            num_workers=1,
        )

        # Transcrever
        segments, info = model.transcribe(
            wav_path,
            beam_size=1 if is_production else 3,
            language=None,  # Detecção automática
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
        )

        # Coletar texto
        text_parts = [seg.text.strip() for seg in segments if seg.text.strip()]

        if not text_parts:
            return "Nenhuma fala detectada no áudio."

        return " ".join(text_parts)

    finally:
        cleanup_files([wav_path])


def cleanup_files(paths: list) -> None:
    """Remove arquivos/diretórios temporários com segurança."""
    for path in paths:
        if not path or not isinstance(path, str):
            continue

        try:
            if os.path.isfile(path):
                os.remove(path)
            elif os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
        except OSError:
            pass


if __name__ == "__main__":
    app = create_app()
    port = int(os.environ.get("PORT", "8000"))
    debug = os.environ.get("FLASK_ENV") != "production"
    app.run(host="0.0.0.0", port=port, debug=debug)
