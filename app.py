import os
import re
import tempfile
import uuid
import shutil
from typing import Optional

from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from yt_dlp import YoutubeDL
from youtube_transcript_api import YouTubeTranscriptApi
from faster_whisper import WhisperModel
import subprocess
import imageio_ffmpeg


ALLOWED_HOSTS = {
    "youtube.com",
    "www.youtube.com",
    "youtu.be",
    "m.youtube.com",
}

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

        # Extrair ID do vídeo
        video_id = extract_video_id(input_url)
        if not video_id:
            flash("Não foi possível extrair o ID do vídeo.", "error")
            return redirect(url_for("index"))

        transcript_text = None
        method_used = None

        # === CAMADA 1: YouTube Transcript API (MAIS RÁPIDO E CONFIÁVEL) ===
        try:
            transcript_text = get_transcript_from_api(video_id)
            method_used = "YouTube Transcript API"
        except Exception as e:
            print(f"Transcript API falhou: {e}")

        # === CAMADA 2: Download + Whisper (SE TRANSCRIPT API FALHAR) ===
        if not transcript_text:
            file_path = None
            try:
                file_path = download_youtube_audio_with_fallbacks(input_url)

                duration = get_audio_duration(file_path)
                if duration > MAX_VIDEO_DURATION:
                    raise ValueError(
                        f"Vídeo muito longo ({int(duration)}s). Máximo: {MAX_VIDEO_DURATION}s"
                    )

                transcript_text = transcribe_with_whisper_local(file_path)
                method_used = "Whisper AI"

            except Exception as exc:
                error_msg = format_error_message(str(exc))
                flash(f"Erro: {error_msg}", "error")
                return redirect(url_for("index"))
            finally:
                cleanup_files([file_path])

        if transcript_text:
            flash(
                f"✅ Transcrição concluída com sucesso! (Método: {method_used})", "success")
            return render_template("index.html", transcript=transcript_text)
        else:
            flash("❌ Não foi possível obter a transcrição por nenhum método.", "error")
            return redirect(url_for("index"))

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
            temp_path = os.path.join(
                tempfile.gettempdir(),
                f"upload_{uuid.uuid4().hex}{file_ext}"
            )
            file.save(temp_path)

            file_size_mb = os.path.getsize(temp_path) / (1024 * 1024)
            if file_size_mb > MAX_VIDEO_SIZE_MB:
                raise ValueError(
                    f"Arquivo muito grande ({file_size_mb:.1f}MB). "
                    f"Máximo: {MAX_VIDEO_SIZE_MB}MB"
                )

            transcript = transcribe_with_whisper_local(temp_path)

            flash("✅ Transcrição concluída com sucesso!", "success")
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
        r"(?:v=|/)([0-9A-Za-z_-]{11}).*",
        r"youtu\.be/([0-9A-Za-z_-]{11})",
        r"embed/([0-9A-Za-z_-]{11})",
        r"shorts/([0-9A-Za-z_-]{11})",
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def get_transcript_from_api(video_id: str) -> Optional[str]:
    """
    CAMADA 1: Obtém transcrição usando YouTube Transcript API.
    Método mais rápido e confiável (não baixa vídeo).
    """
    try:
        # Tentar obter em português primeiro
        transcript_list = YouTubeTranscriptApi.get_transcript(
            video_id,
            languages=['pt', 'pt-BR', 'en']
        )

        # Juntar todos os textos
        full_text = " ".join([entry['text'] for entry in transcript_list])

        if full_text.strip():
            return full_text.strip()

        return None

    except Exception as e:
        # Se não houver legendas, retornar None para tentar próximo método
        print(f"YouTube Transcript API falhou: {e}")
        return None


def download_youtube_audio_with_fallbacks(url: str) -> str:
    """
    CAMADA 2: Download com múltiplas estratégias de player client.
    """
    temp_dir = tempfile.mkdtemp(prefix="yt_audio_")
    unique = uuid.uuid4().hex[:8]
    output_path = os.path.join(temp_dir, f"audio_{unique}.%(ext)s")

    # Lista de configurações para tentar em sequência
    player_configs = [
        # Configuração 1: Android client (mais confiável atualmente)
        {
            "name": "android",
            "player_client": ["android"],
            "player_skip": ["webpage"],
        },
        # Configuração 2: iOS client
        {
            "name": "ios",
            "player_client": ["ios"],
            "player_skip": ["webpage"],
        },
        # Configuração 3: Web client com embed
        {
            "name": "web_embed",
            "player_client": ["web"],
            "player_skip": [],
        },
        # Configuração 4: Android + Web
        {
            "name": "android_web",
            "player_client": ["android", "web"],
            "player_skip": ["configs"],
        },
    ]

    last_error = None

    for config in player_configs:
        try:
            print(f"Tentando com player: {config['name']}")

            ydl_opts = {
                "outtmpl": output_path,
                "format": "bestaudio[ext=m4a]/bestaudio/best",

                # Headers realistas
                "http_headers": {
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/131.0.0.0 Safari/537.36"
                    ),
                    "Accept-Language": "en-US,en;q=0.9",
                    "Accept": "*/*",
                    "Referer": "https://www.youtube.com/",
                },

                # Configuração específica do player
                "extractor_args": {
                    "youtube": {
                        "player_client": config["player_client"],
                        "player_skip": config["player_skip"],
                    }
                },

                # Cookies (se disponível)
                "cookiefile": os.environ.get("YOUTUBE_COOKIES_PATH"),

                # Rede
                "socket_timeout": 30,
                "retries": 2,
                "fragment_retries": 2,

                # Geral
                "quiet": False,  # Ver logs para debug
                "no_warnings": False,
                "ignoreerrors": False,
                "nocheckcertificate": True,

                # Limites
                "max_filesize": MAX_VIDEO_SIZE_MB * 1024 * 1024,
            }

            with YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)

                downloaded_file = find_downloaded_file(temp_dir)
                if downloaded_file:
                    print(f"✅ Sucesso com player: {config['name']}")
                    return downloaded_file

        except Exception as e:
            last_error = str(e)
            print(f"❌ Falhou com player {config['name']}: {e}")
            continue

    # Se todos falharam, limpar e lançar erro
    cleanup_files([temp_dir])
    raise RuntimeError(
        f"Falha em todos os métodos de download. Último erro: {last_error}"
    )


def find_downloaded_file(directory: str) -> Optional[str]:
    """Localiza o arquivo de áudio baixado."""
    audio_extensions = ('.m4a', '.mp3', '.opus', '.webm', '.ogg', '.wav')

    try:
        for filename in os.listdir(directory):
            if filename.endswith(audio_extensions):
                return os.path.join(directory, filename)
    except Exception:
        pass

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
        timeout=300
    )

    if proc.returncode != 0 or not os.path.exists(wav_path):
        stderr = proc.stderr.decode('utf-8', errors='ignore')
        raise RuntimeError(f"Falha ao extrair áudio: {stderr[:200]}")

    return wav_path


def transcribe_with_whisper_local(file_path: str) -> str:
    """Transcreve áudio usando Whisper otimizado."""
    wav_path = None

    try:
        wav_path = extract_wav_with_bundled_ffmpeg(file_path)

        is_production = os.environ.get("FLASK_ENV") == "production"
        model_size = "tiny" if is_production else os.environ.get(
            "WHISPER_MODEL", "base"
        )

        model = WhisperModel(
            model_size,
            device="cpu",
            compute_type="int8",
            num_workers=1,
        )

        segments, info = model.transcribe(
            wav_path,
            beam_size=1 if is_production else 3,
            language=None,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
        )

        text_parts = [seg.text.strip() for seg in segments if seg.text.strip()]

        if not text_parts:
            return "⚠️ Nenhuma fala detectada no áudio."

        return " ".join(text_parts)

    finally:
        cleanup_files([wav_path])


def format_error_message(error: str) -> str:
    """Formata mensagens de erro para serem amigáveis ao usuário."""
    if "Sign in to confirm" in error or "bot" in error.lower():
        return (
            "YouTube bloqueou a requisição. "
            "Tente: (1) Aguardar 10-15 min, (2) Usar outra URL, (3) Upload direto do arquivo."
        )
    elif "Video unavailable" in error:
        return "Vídeo indisponível, privado ou restrito por região."
    elif "player response" in error.lower():
        return (
            "Não foi possível extrair informações do vídeo. "
            "O vídeo pode ter legendas desabilitadas. Tente fazer upload do arquivo."
        )
    elif "Requested format not available" in error:
        return "Formato de vídeo não disponível para download."
    else:
        return error


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
