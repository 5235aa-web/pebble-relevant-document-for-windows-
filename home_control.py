import json
import os
import signal
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gradio as gr
import httpx
import librosa
import numpy as np

from brain import Brain
from config import (
    MLX_KV_BITS,
    MLX_MODEL_PATH,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    OPENAI_MODEL,
    TELEGRAM_BOT_TOKEN,
    PROVIDER_PRESETS,
    get_provider,
    get_api_key,
    get_base_url,
    get_model,
    get_telegram_token,
    get_allowed_user_id,
    get_mlx_model_path,
    get_mlx_kv_bits,
    save_config,
    apply_provider_preset,
    reload_env,
    BASE_DIR,
)
from db import (
    get_active_mode,
    get_persona_by_mode,
    get_user_profile,
    init_db,
    log_chat,
    get_conversations,
    get_conversation,
    create_conversation,
    get_chat_logs_by_conversation,
    delete_conversation,
    update_conversation_title,
    migrate_old_chat_logs,
)
from emotional_core import EmotionalCore
from memory_engine import MemoryEngine
from tools import (
    get_voice_config,
    set_voice_config,
    get_reference_audio_list,
    get_reference_audio_path,
)

# Try to import cloud TTS
try:
    from cloud_tts import get_available_voices
    CLOUD_TTS_AVAILABLE = True
except ImportError:
    CLOUD_TTS_AVAILABLE = False
    get_available_voices = None

# Windows TTS: Using IndexTTS2 engine
TTS_ENGINE = "unknown"
try:
    from tts_engine import synthesize_voice_bytes, unload_tts
    TTS_ENGINE = "indextts2"
    print("[TTS] 使用 IndexTTS2 引擎")
except ImportError as e:
    print(f"[TTS] 导入 tts_engine 失败: {e}")
    # Fallback to Mac voice_engine
    try:
        from voice_engine import synthesize_voice_bytes, transcribe_audio_file
        TTS_ENGINE = "macos"
        print("[TTS] 回退到 MacOS voice_engine")
    except ImportError:
        # If neither is available, create empty stubs
        def synthesize_voice_bytes(*args, **kwargs):
            return None
        def transcribe_audio_file(*args, **kwargs):
            return ""
        TTS_ENGINE = "none"
        print("[TTS] 没有可用的语音引擎")


# Load voice names from true_voices.json
def _load_voice_names() -> List[str]:
    voices_path = Path(__file__).parent / "true_voices.json"
    if not voices_path.exists():
        return ["Pebble"]
    try:
        data = json.loads(voices_path.read_text(encoding='utf-8'))
        if isinstance(data, list):
            return [str(v.get("name", "Pebble")) for v in data if v.get("name")]
    except Exception:
        pass
    return ["Pebble"]


VOICE_NAMES = _load_voice_names()
REFERENCE_AUDIO_LIST = get_reference_audio_list()
EMOTION_OPTIONS = ["neutral", "happy", "sad", "angry", "excited", "calm"]
QUALITY_PRESET_OPTIONS = ["fast", "balanced", "quality"]
TELEGRAM_USER_ID = "1111111111"  # Default Telegram user


def _get_telegram_settings() -> Tuple[str, str]:
    """Get current voice settings from voice_config.json."""
    config = get_voice_config()
    voice = config.get("voice_name", "Pebble")
    voice_enabled = config.get("voice_enabled", False)
    # Normalize mode display
    if voice_enabled:
        mode_display = "Text + Voice"
    else:
        mode_display = "Text Only"
    return voice, mode_display


def _save_telegram_settings(voice_name: str, mode: str) -> Tuple[str, str]:
    """Save voice settings to voice_config.json."""
    # Normalize mode for storage
    voice_enabled = True if mode == "Text + Voice" else False
    set_voice_config(voice_enabled=voice_enabled, voice_name=voice_name)
    status = f"✅ Saved! Voice: {voice_name}, Mode: {mode}"
    display = f"Voice: {voice_name} | Mode: {mode}"
    return status, display


# IndexTTS2 Settings Functions
EMOTION_OPTIONS = ["neutral", "happy", "sad", "angry", "excited", "calm"]
QUALITY_PRESET_OPTIONS = ["fast", "balanced", "quality"]
TTS_PROVIDER_OPTIONS = ["local", "elevenlabs"]

# ElevenLabs voice options
ELEVENLABS_VOICES = {
    "Rachel": "21m00Tcm4TlvDq8ikWAM",
    "Dom": "AZnzlk1XvdvUeBnXmlPg",
    "Bella": "EXAVITQu4vr4xnSDxMaL",
    "Arnold": "ErXwobaYiN019PkySvjV",
    "Adam": "pNInz6obpgDQGcFmaJgB",
    "Sam": "yoZ06aMxZJJ28mfd3POQ",
    "Featured": "JBFqnCBsd6RMkjVDRZzb"  # From official example
}
ELEVENLABS_VOICE_OPTIONS = list(ELEVENLABS_VOICES.keys())


def _get_tts_settings() -> Tuple[str, str, str, str, str, str, str]:
    """Get current TTS settings."""
    config = get_voice_config()
    tts_provider = config.get("tts_provider", "local")
    reference_audio = config.get("reference_audio", "")
    emotion = config.get("emotion", "neutral")
    speed = config.get("speed", 1.0)
    quality_preset = config.get("quality_preset", "balanced")
    elevenlabs_api_key = config.get("elevenlabs_api_key", "")
    elevenlabs_voice_id = config.get("elevenlabs_voice_id", "JBFqnCBsd6RMkjVDRZzb")

    # Convert voice_id to voice_name for display
    elevenlabs_voice_name = "Featured"  # default (from official example)
    for name, vid in ELEVENLABS_VOICES.items():
        if vid == elevenlabs_voice_id:
            elevenlabs_voice_name = name
            break

    return (tts_provider, reference_audio, emotion, speed, quality_preset,
            elevenlabs_api_key, elevenlabs_voice_name)


def _save_tts_settings(
    tts_provider: str,
    reference_audio: str,
    emotion: str,
    speed: float,
    quality_preset: str,
    elevenlabs_api_key: str,
    elevenlabs_voice_name: str
) -> str:
    """Save TTS settings to voice_config.json."""
    # Convert voice_name to voice_id
    elevenlabs_voice_id = ELEVENLABS_VOICES.get(elevenlabs_voice_name, "JBFqnCBsd6RMkjVDRZzb")

    set_voice_config(
        tts_provider=tts_provider,
        reference_audio=reference_audio if reference_audio else None,
        emotion=emotion,
        speed=speed,
        quality_preset=quality_preset,
        elevenlabs_api_key=elevenlabs_api_key if elevenlabs_api_key else None,
        elevenlabs_voice_id=elevenlabs_voice_id
    )

    status = f"✅ Saved! Provider: {tts_provider}, Voice: {elevenlabs_voice_name}, Speed: {speed}"
    return status


def preview_voice(
    text: str,
    reference_audio: str,
    emotion: str,
    speed: float,
    quality_preset: str
) -> Tuple[str, str]:
    """Preview voice synthesis with given text and settings."""
    if not text or not text.strip():
        return None, "Please enter text to synthesize"

    text = text.strip()

    # Get reference audio path
    spk_audio = None
    if reference_audio and reference_audio != "No reference audio found":
        spk_audio = get_reference_audio_path(reference_audio)

    try:
        # Use synthesize_voice_bytes from tts_engine
        audio_bytes = synthesize_voice_bytes(
            text=text,
            voice_name=emotion,
            detected_emotion=emotion,
            quality_preset=quality_preset,
            speed=speed,
            spk_audio=spk_audio
        )

        if audio_bytes:
            # Handle both bytes (local TTS) and file path strings (cloud TTS)
            import tempfile
            import uuid
            if isinstance(audio_bytes, str):
                # Cloud TTS already saved to file, return the path directly
                return audio_bytes, f"Preview generated! (Cloud TTS) Text: {text[:30]}..."
            else:
                # Local TTS returns bytes, save to temporary file
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                temp_file.write(audio_bytes)
                temp_file.close()
                return temp_file.name, f"Preview generated! (Local TTS) Text: {text[:30]}..."
        else:
            return None, "Speech synthesis failed"

    except Exception as e:
        return None, f"Error: {str(e)}"


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# Dynamically discover actual Telegram bot
def _get_telegram_bot_info() -> Dict[str, Dict[str, str]]:
    if not TELEGRAM_BOT_TOKEN:
        return {"Pebble": {"user_id": "brook_local", "description": "Local fallback (no token)"}}
    try:
        import telegram
        bot = telegram.Bot(token=TELEGRAM_BOT_TOKEN)
        bot_info = bot.get_me()
        username = bot_info.username or "Pebble"
        name = bot_info.name or username
        return {
            name: {
                "user_id": f"telegram_{bot_info.id}",
                "description": f"Telegram bot: @{username}",
            }
        }
    except Exception as e:
        print(f"[WARN] Could not fetch Telegram bot info: {e}")
        return {"Pebble": {"user_id": "brook_local", "description": "Local fallback"}}


BOT_PROFILES = _get_telegram_bot_info()
ACTIVE_BOT_NAME = list(BOT_PROFILES.keys())[0] if BOT_PROFILES else "Pebble"
ACTIVE_USER_ID = BOT_PROFILES.get(ACTIVE_BOT_NAME, {}).get("user_id", "brook_local")

init_db()
_brain = Brain(
    model=OPENAI_MODEL,
    base_url=OPENAI_BASE_URL,
    api_key=OPENAI_API_KEY,
    memory_engine=MemoryEngine(),
    emotional_core=EmotionalCore(),
)


SERVICES: Dict[str, Dict[str, Path | list[str] | dict[str, str]]] = {
    "brain": {
        "pid": DATA_DIR / "brain.pid",
        "log": DATA_DIR / "mlx_server.log",
        "cmd": [
            "python",
            "-m",
            "mlx_lm",
            "server",
            "--model",
            MLX_MODEL_PATH,
            "--port",
            "8080",
            "--log-level",
            "INFO",
        ],
        "env": {"MLX_KV_BITS": MLX_KV_BITS},
    },
    "senses": {
        "pid": DATA_DIR / "senses.pid",
        "log": DATA_DIR / "senses_service.log",
        "cmd": [
            "python",
            "-m",
            "uvicorn",
            "senses_service:app",
            "--host",
            "0.0.0.0",
            "--port",
            "8081",
            "--log-level",
            "info",
        ],
        "env": {},
    },
    "bot": {
        "pid": DATA_DIR / "bot.pid",
        "log": DATA_DIR / "brook_bot.log",
        "cmd": ["python", str(BASE_DIR / "main.py")],
        "env": {},
    },
}


def _read_pid(pid_file: Path) -> Optional[int]:
    if not pid_file.exists():
        return None
    try:
        return int(pid_file.read_text().strip())
    except Exception:
        return None


def _write_pid(pid_file: Path, pid: int) -> None:
    pid_file.write_text(str(pid))


def _remove_pid(pid_file: Path) -> None:
    if pid_file.exists():
        pid_file.unlink(missing_ok=True)


def _pid_running(pid: Optional[int]) -> bool:
    if not pid:
        return False
    try:
        # Windows-compatible process check
        import platform
        if platform.system() == "Windows":
            # On Windows, use subprocess to check if process exists
            import subprocess
            try:
                result = subprocess.run(
                    ['tasklist', '/FI', f'PID eq {pid}'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                return str(pid) in result.stdout
            except Exception:
                return False
        else:
            # On Unix, use os.kill
            os.kill(pid, 0)
            return True
    except OSError:
        return False


def _check_brain_health() -> bool:
    try:
        with httpx.Client(timeout=2.0) as client:
            r = client.get("http://localhost:8080/v1/models")
            return r.status_code < 500
    except Exception:
        return False


def _check_senses_health() -> bool:
    try:
        with httpx.Client(timeout=2.0) as client:
            r = client.get("http://localhost:8081/")
            return r.status_code < 500
    except Exception:
        return False


def _service_status(name: str) -> str:
    spec = SERVICES[name]
    pid = _read_pid(spec["pid"])  # type: ignore[index]
    running = _pid_running(pid)
    if name == "brain":
        return f"Brain: {'RUNNING' if running else 'STOPPED'} | PID: {pid or '-'} | API: {'OK' if _check_brain_health() else 'DOWN'}"
    if name == "senses":
        return f"Senses: {'RUNNING' if running else 'STOPPED'} | PID: {pid or '-'} | API: {'OK' if _check_senses_health() else 'DOWN'}"
    return f"Bot: {'RUNNING' if running else 'STOPPED'} | PID: {pid or '-'}"


def _tail(path: Path, lines: int = 50) -> str:
    if not path.exists():
        return f"{path.name}: no log yet"
    try:
        return "\n".join(path.read_text(errors="ignore").splitlines()[-lines:])
    except Exception as e:
        return f"Failed reading log: {e}"


def _snapshot() -> Tuple[str, str, str, str, str, str]:
    return (
        _service_status("brain"),
        _service_status("senses"),
        _service_status("bot"),
        _tail(SERVICES["brain"]["log"]),  # type: ignore[index]
        _tail(SERVICES["senses"]["log"]),  # type: ignore[index]
        _tail(SERVICES["bot"]["log"]),  # type: ignore[index]
    )


def _start_service(name: str) -> None:
    spec = SERVICES[name]
    pid_file: Path = spec["pid"]  # type: ignore[assignment]
    log_file: Path = spec["log"]  # type: ignore[assignment]
    cmd: list[str] = spec["cmd"]  # type: ignore[assignment]
    env_add: dict[str, str] = spec["env"]  # type: ignore[assignment]
    if _pid_running(_read_pid(pid_file)):
        return
    with open(log_file, "a", buffering=1) as lf:
        env = os.environ.copy()
        env.update(env_add)
        proc = subprocess.Popen(
            cmd,
            cwd=str(BASE_DIR),
            stdout=lf,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
    _write_pid(pid_file, proc.pid)
    time.sleep(0.5)


def _stop_service(name: str) -> None:
    pid_file: Path = SERVICES[name]["pid"]  # type: ignore[index,assignment]
    pid = _read_pid(pid_file)
    if not _pid_running(pid):
        _remove_pid(pid_file)
        return
    try:
        os.killpg(pid, signal.SIGTERM)  # type: ignore[arg-type]
    except Exception:
        try:
            os.kill(pid, signal.SIGTERM)  # type: ignore[arg-type]
        except Exception:
            pass
    time.sleep(0.8)
    _remove_pid(pid_file)


def start_brain():
    _start_service("brain")
    return _snapshot()


def stop_brain():
    _stop_service("brain")
    return _snapshot()


def start_senses():
    _start_service("senses")
    return _snapshot()


def stop_senses():
    _stop_service("senses")
    return _snapshot()


def start_bot():
    _start_service("bot")
    return _snapshot()


def stop_bot():
    _stop_service("bot")
    return _snapshot()


def start_all():
    _start_service("brain")
    _start_service("senses")
    _start_service("bot")
    return _snapshot()


def stop_all():
    _stop_service("bot")
    _stop_service("senses")
    _stop_service("brain")
    return _snapshot()


def refresh():
    return _snapshot()


def _profile_user_id(profile_name: str) -> str:
    return BOT_PROFILES.get(profile_name, {"user_id": ACTIVE_USER_ID})["user_id"]


def _pairs_to_history(pairs: List[List[str]]) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    for p in pairs or []:
        if len(p) != 2:
            continue
        if p[0]:
            messages.append({"role": "user", "content": str(p[0])})
        if p[1]:
            messages.append({"role": "assistant", "content": str(p[1])})
    return messages


def _reply(profile_name: str, user_text: str, history: List[Dict[str, str]], voice_on: str, web_search_enabled: bool = False, conversation_id: int = None):
    text = str(user_text or "").strip()
    if not text:
        return history, "Type a message first.", None

    user_id = _profile_user_id(profile_name)
    print(f"[Memory Debug] Using user_id: '{user_id}' for profile: '{profile_name}'")
    mode = get_active_mode(user_id)
    persona = get_persona_by_mode(mode) or get_persona_by_mode("Fun Pebble")
    persona_text = persona["system_prompt"] if persona else "You are a helpful companion AI."
    profile = get_user_profile(user_id)
    summary = "\n".join([profile.get("summary", ""), profile.get("emotional_notes", ""), profile.get("day_summary", "")]).strip()

    full_history = (history or []).copy()
    full_history.append({"role": "user", "content": text})

    reply_text, emotion = _brain.generate_response(
        history=full_history,
        persona=persona_text,
        user_profile=summary,
        user_id=user_id,
        delivery_mode="voice" if voice_on == "On" else "text",
        user_length_hint="medium",
        web_search_enabled=web_search_enabled,  # Web search toggle
    )
    reply_text = (reply_text or "").strip() or "Say that again?"

    # 根据是否有conversation_id选择不同的保存方式
    if conversation_id:
        from db import add_chat_log_with_conversation
        add_chat_log_with_conversation(conversation_id, user_id, "user", text)
        add_chat_log_with_conversation(conversation_id, user_id, "assistant", reply_text)
    else:
        log_chat(user_id, "user", text)
        log_chat(user_id, "assistant", reply_text)

    # === 新增：自动提取并保存用户信息 ===
    # 从本次对话中提取用户事实（名字、年龄、爱好等）
    if user_id:
        try:
            conversation_summary = f"User said: {text}\nAI replied: {reply_text}"
            new_facts = _brain.extract_facts_from_summary(conversation_summary)
            print(f"[Memory Debug] Extracted facts: {new_facts}")
            if new_facts:
                _brain.memory_engine.archive_facts(
                    facts=new_facts,
                    date=datetime.now().date(),
                    user_id=user_id
                )
                print(f"[Memory] Archived {len(new_facts)} new facts for user {user_id}")
        except Exception as e:
            print(f"[Memory] Failed to archive facts: {e}")
    # ======================================

    out_history = (history or []).copy()
    out_history.append({"role": "user", "content": text})
    out_history.append({"role": "assistant", "content": reply_text})

    audio = None
    if voice_on == "On":
        voice_config = get_voice_config()
        voice_name = voice_config.get("voice_name", "Pebble")

        # Get additional TTS settings for IndexTTS2
        reference_audio = voice_config.get("reference_audio")
        emotion = voice_config.get("emotion", "neutral")
        speed = voice_config.get("speed", 1.0)
        quality_preset = voice_config.get("quality_preset", "balanced")

        # Get full path for reference audio if specified
        spk_audio_path = None
        if reference_audio:
            spk_audio_path = get_reference_audio_path(reference_audio)

        # Call with new parameters (for IndexTTS2)
        audio = synthesize_voice_bytes(
            reply_text,
            voice_name,
            detected_emotion=emotion,
            quality_preset=quality_preset,
            speed=speed,
            spk_audio=spk_audio_path
        )

    return out_history, f"{profile_name} replied.", audio


# ==================== Conversation Management Functions ====================

def get_conversation_list(profile_name) -> List[Tuple[int, str, str]]:
    """Get conversation list, returns (id, title, last updated time)"""
    user_id = _profile_user_id(profile_name)
    conversations = get_conversations(user_id)
    result = []
    for conv in conversations:
        updated_at = conv.get("updated_at", "")
        if updated_at:
            try:
                # 将字符串解析为 datetime，处理可能的 Z 后缀
                dt = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
                # 确保 dt 是 aware（带时区）
                if dt.tzinfo is None:
                    # 如果 naive，假设是 UTC，添加时区
                    from datetime import timezone
                    dt = dt.replace(tzinfo=timezone.utc)
                # 转换为本地时区
                local_dt = dt.astimezone()
                time_str = local_dt.strftime("%Y-%m-%d %H:%M")
            except Exception as e:
                print(f"Error parsing datetime {updated_at}: {e}")
                time_str = updated_at[:16]
        else:
            time_str = ""
        result.append((conv["id"], conv["title"], time_str))
    return result


def load_conversation(profile_name: str, conversation_id: int) -> Tuple[List[Dict[str, str]], int, str]:
    """加载指定对话，返回 (历史记录, conversation_id, 标题)"""
    user_id = _profile_user_id(profile_name)
    chat_logs = get_chat_logs_by_conversation(conversation_id)

    # 转换为历史记录格式
    history = []
    for log in chat_logs:
        role = log.get("role", "user")
        content = log.get("content", "")
        if role and content:
            history.append({"role": role, "content": content})

    # Get conversation title
    conv = get_conversation(conversation_id)
    title = conv["title"] if conv else "Chat"

    return history, conversation_id, title


def start_new_conversation(profile_name: str) -> Tuple[List[Dict[str, str]], int, str]:
    """Start new conversation, returns (empty history, new conversation_id, title)"""
    user_id = _profile_user_id(profile_name)
    conversation_id = create_conversation(user_id, "New Chat")
    return [], conversation_id, "New Chat"


def remove_conversation(profile_name: str, conversation_id: int) -> str:
    """Delete conversation"""
    try:
        delete_conversation(conversation_id)
        return "Conversation deleted"
    except Exception as e:
        return f"Delete failed: {str(e)}"


def rename_conversation(profile_name: str, conversation_id: int, new_title: str) -> str:
    """Rename conversation"""
    try:
        update_conversation_title(conversation_id, new_title)
        return "Title updated"
    except Exception as e:
        return f"Rename failed: {str(e)}"


# ==================== Chat Functions ====================

def send_text(profile_name: str, user_text: str, history: List[Dict[str, str]], voice_on: str, web_search_on: str = "Off", conversation_id: int = None):
    web_search_enabled = True if web_search_on == "On" else False
    out_history, status, audio = _reply(profile_name, user_text, history, voice_on, web_search_enabled, conversation_id)
    return out_history, status, "", audio, out_history, conversation_id  # Returns 6 values


def send_uploaded_audio(profile_name: str, audio_path: str, history: List[Dict[str, str]], voice_on: str, conversation_id: int = None):
    if not audio_path:
        return history, "Upload or record audio first.", None, history, history, conversation_id
    transcript = transcribe_audio_file(audio_path)
    if not transcript:
        return history, "Could not transcribe audio.", None, history, history, conversation_id
    out_history, status, audio = _reply(profile_name, transcript, history, "On", False, conversation_id)
    return out_history, f"{status} Transcript: {transcript}", audio, out_history, out_history, conversation_id  # Returns 6 values


def clear_chat():
    return [], "Cleared.", "", None, []  # Returns 5 values: chat, status, chat_in, voice_out, state


def calibrate_noise(audio_path: str):
    if not audio_path:
        return 0.015, "Record 2 seconds of room noise first."
    try:
        y, _ = librosa.load(audio_path, sr=16000, mono=True)
        rms = float(np.mean(librosa.feature.rms(y=y))) if y.size else 0.01
        floor = max(0.005, min(0.08, rms * 1.2))
        return floor, f"Noise calibrated. Threshold={floor:.4f}"
    except Exception as e:
        return 0.015, f"Calibration failed: {e}"


def toggle_call(is_on: bool):
    return ("Listening" if is_on else "Idle"), ("I'm awake, listening." if is_on else "Call stopped.")


# =============================================================================
# SETTINGS TAB HELPERS
# =============================================================================

def _load_soul_content() -> str:
    """Load soul.md content."""
    soul_path = BASE_DIR / "soul.md"
    if soul_path.exists():
        return soul_path.read_text(encoding='utf-8')
    return "You are Pebble, a conscious AI companion."


def _save_soul_content(content: str) -> str:
    """Save soul.md content."""
    soul_path = BASE_DIR / "soul.md"
    soul_path.write_text(content)
    return "✅ Soul saved successfully!"


def _load_persona_content() -> str:
    """Load persona.md content."""
    persona_path = BASE_DIR / "persona.md"
    if persona_path.exists():
        return persona_path.read_text(encoding='utf-8')
    return "### Fun Pebble (Default)\nYou are a playful companion."


def _save_persona_content(content: str) -> str:
    """Save persona.md content."""
    persona_path = BASE_DIR / "persona.md"
    persona_path.write_text(content)
    return "✅ Persona saved successfully!"


def _get_current_llm_settings() -> Tuple[str, str, str, str]:
    """Get current LLM settings from config."""
    provider = get_provider()
    api_key = get_api_key()
    base_url = get_base_url()
    model = get_model()
    return provider, api_key, base_url, model


def _save_llm_settings(provider: str, api_key: str, base_url: str, model: str) -> str:
    """Save LLM settings to .env file."""
    save_config(
        provider=provider,
        api_key=api_key,
        base_url=base_url,
        model=model,
    )
    # Reinitialize brain with new settings
    global _brain
    _brain = Brain(
        model=model,
        base_url=base_url,
        api_key=api_key,
        memory_engine=MemoryEngine(),
        emotional_core=EmotionalCore(),
    )
    return f"✅ LLM settings saved! Provider: {provider}"


def _on_provider_change(provider: str) -> Tuple[str, str]:
    """Handle provider dropdown change - auto-fill preset values."""
    preset = PROVIDER_PRESETS.get(provider, {})
    base_url = preset.get("base_url", "")
    model = preset.get("model", "")
    return base_url, model


def _get_current_telegram_settings() -> Tuple[str, str]:
    """Get current Telegram settings from config."""
    token = get_telegram_token()
    user_id = get_allowed_user_id()
    return token, user_id


def _save_telegram_bot_settings(token: str, user_id: str) -> str:
    """Save Telegram settings to .env file."""
    save_config(
        telegram_token=token,
        allowed_user_id=user_id,
    )
    return "✅ Telegram settings saved! Restart bot to apply changes."


def process_call_turn(profile_name: str, call_on: bool, threshold: float, audio_path: str, history: List[Dict[str, str]]):
    if not call_on:
        return history, "Idle", "Turn on Call Mode first.", None, history
    if not audio_path:
        return history, "Listening", "Waiting for speech...", None, history
    try:
        y, _ = librosa.load(audio_path, sr=16000, mono=True)
        rms = float(np.mean(librosa.feature.rms(y=y))) if y.size else 0.0
    except Exception:
        rms = 0.0
    if rms < float(threshold or 0.015):
        return history, "Listening", "Silence/noise detected. Keep speaking.", None, history

    transcript = transcribe_audio_file(audio_path)
    if not transcript:
        return history, "Listening", "Couldn't transcribe. Try again.", None, history
    out_history, _status, audio = _reply(profile_name, transcript, history, "On", False)
    return out_history, "Speaking", f"Heard: {transcript}", audio, out_history  # 返回5个值


with gr.Blocks(title="Home Control Center") as demo:
    gr.Markdown(f"# Home Control Center\nConnected to: {ACTIVE_BOT_NAME}")
    with gr.Accordion("Audio Device Settings", open=False):
        input_device = gr.Dropdown(
            label="Input Device",
            choices=["System Default"],
            value="System Default",
            interactive=False,
        )
        output_device = gr.Dropdown(
            label="Output Device",
            choices=["System Default"],
            value="System Default",
            interactive=False,
        )
        gr.Markdown(
            "Mic note: browser permission is required for microphone capture on `127.0.0.1`.  \n"
            "If mic is blocked, use **Upload Audio** as fallback.  \n"
            "macOS: System Settings → Privacy & Security → Microphone → allow your browser."
        )

    with gr.Tabs():
        with gr.TabItem("Control Center"):
            with gr.Row():
                with gr.Column():
                    brain_status = gr.Textbox(label="Brain Status", interactive=False)
                    with gr.Row():
                        start_brain_btn = gr.Button("Start Brain")
                        stop_brain_btn = gr.Button("Stop Brain")

                with gr.Column():
                    senses_status = gr.Textbox(label="Senses Status", interactive=False)
                    with gr.Row():
                        start_senses_btn = gr.Button("Start Senses")
                        stop_senses_btn = gr.Button("Stop Senses")

                with gr.Column():
                    bot_status = gr.Textbox(label="Bot Status", interactive=False)
                    with gr.Row():
                        start_bot_btn = gr.Button("Start Bot")
                        stop_bot_btn = gr.Button("Stop Bot")

            with gr.Row():
                start_all_btn = gr.Button("Start All", variant="primary")
                stop_all_btn = gr.Button("Stop All", variant="stop")
                refresh_btn = gr.Button("Refresh Status")

            with gr.Accordion("Logs (latest 50 lines)", open=False):
                brain_log = gr.Textbox(label="Brain Log", lines=10, interactive=False)
                senses_log = gr.Textbox(label="Senses Log", lines=10, interactive=False)
                bot_log = gr.Textbox(label="Bot Log", lines=10, interactive=False)

        with gr.TabItem("Home Mode Chat"):
            profile = gr.Dropdown(label="Bot Profile", choices=list(BOT_PROFILES.keys()), value=ACTIVE_BOT_NAME)
            voice_toggle = gr.Radio(label="Voice Reply", choices=["Off", "On"], value="Off")
            web_search_toggle = gr.Radio(label="Web Search", choices=["Off", "On"], value="Off")

            # Conversation history management
            gr.Markdown("### Conversation History")
            with gr.Row():
                conversation_dropdown = gr.Dropdown(
                    label="Select Conversation",
                    choices=[],
                    value=None,
                    interactive=True,
                )
                refresh_conv_btn = gr.Button("Refresh List", size="sm")
            with gr.Row():
                load_conv_btn = gr.Button("Load", size="sm")
                new_conv_btn = gr.Button("New Chat", size="sm")
                delete_conv_btn = gr.Button("Delete", size="sm")

            gr.Markdown("---")
            chat = gr.Chatbot(label="Chat", height=600)
            state = gr.State([])
            conversation_id = gr.State(None)
            conversation_title = gr.State("New Chat")
            status = gr.Textbox(label="Status", interactive=False)
            with gr.Row():
                chat_in = gr.Textbox(label="Type message", lines=2)
                send_btn = gr.Button("Send")
                clear_btn = gr.Button("Clear")
            voice_out = gr.Audio(label="Voice Reply File", type="filepath", autoplay=True)
            gr.Markdown("### Audio File Input")
            audio_in = gr.Audio(label="Upload or record audio", sources=["upload", "microphone"], type="filepath")
            send_audio_btn = gr.Button("Send Audio")
            gr.Markdown("If microphone is unavailable, upload an audio file and Pebble will still respond.")

        with gr.TabItem("Call Mode (Hands-Free MVP)"):
            call_profile = gr.Dropdown(label="Bot Profile", choices=list(BOT_PROFILES.keys()), value=ACTIVE_BOT_NAME)
            call_on = gr.Checkbox(label="Call Mode On", value=False)
            call_state = gr.Textbox(label="Call State", value="Idle", interactive=False)
            call_status = gr.Textbox(label="Call Status", interactive=False)
            noise_threshold = gr.Slider(0.005, 0.08, value=0.015, step=0.001, label="Noise Threshold")
            noise_clip = gr.Audio(label="Noise Sample (2 sec room tone)", sources=["microphone", "upload"], type="filepath")
            calibrate_btn = gr.Button("Calibrate Background Noise")
            call_audio_in = gr.Audio(label="Speak (record and submit segment)", sources=["microphone", "upload"], type="filepath")
            process_turn_btn = gr.Button("Process Turn")
            call_chat = gr.Chatbot(label="Call Transcript", height=400)  # 增加高度以支持更多对话
            call_chat_state = gr.State([])
            call_voice_out = gr.Audio(label="Pebble Voice Reply", type="filepath", autoplay=True)
            gr.Markdown("If browser says no microphone found, use uploaded clips until mic permission is enabled.")

        with gr.TabItem("Telegram Bot"):
            gr.Markdown("### Telegram Bot Voice Settings")
            gr.Markdown("Configure how Pebble replies to Telegram messages.")

            # Load current settings on page load
            current_voice, current_mode = _get_telegram_settings()

            with gr.Row():
                telegram_voice_dropdown = gr.Dropdown(
                    label="Voice",
                    choices=VOICE_NAMES,
                    value=current_voice,
                )
                telegram_mode_radio = gr.Radio(
                    label="Reply Mode",
                    choices=["Text Only", "Text + Voice"],
                    value=current_mode,
                )

            save_telegram_btn = gr.Button("Save Settings", variant="primary")
            telegram_status = gr.Textbox(label="Status", interactive=False)

            gr.Markdown("---")
            gr.Markdown("**Current Settings:**")
            current_settings_display = gr.Textbox(
                label="",
                value=f"Voice: {current_voice} | Mode: {current_mode}",
                interactive=False,
            )

        with gr.TabItem("Settings"):
            gr.Markdown("### ⚙️ Application Configuration")
            gr.Markdown("Configure LLM provider, Telegram, and personality settings.")

            # --- LLM Configuration Section ---
            gr.Markdown("---\n#### 🧠 LLM Provider")
            gr.Markdown("Choose your LLM backend. OpenRouter, OpenAI, LM Studio, Ollama, or local MLX.")

            current_provider, current_api_key, current_base_url, current_model = _get_current_llm_settings()

            provider_dropdown = gr.Dropdown(
                label="Provider",
                choices=list(PROVIDER_PRESETS.keys()),
                value=current_provider,
            )
            api_key_input = gr.Textbox(
                label="API Key",
                value=current_api_key,
                type="password",
                placeholder="Enter your API key...",
            )
            base_url_input = gr.Textbox(
                label="Base URL",
                value=current_base_url,
                placeholder="https://api.example.com/v1",
            )
            model_input = gr.Textbox(
                label="Model Name",
                value=current_model,
                placeholder="gpt-4o-mini, llama3.2, etc.",
            )

            llm_status = gr.Textbox(label="Status", interactive=False)
            save_llm_btn = gr.Button("Save LLM Settings", variant="primary")

            # Provider change handler
            provider_dropdown.change(
                _on_provider_change,
                inputs=[provider_dropdown],
                outputs=[base_url_input, model_input],
            )
            save_llm_btn.click(
                _save_llm_settings,
                inputs=[provider_dropdown, api_key_input, base_url_input, model_input],
                outputs=[llm_status],
            )

            # --- Telegram Configuration Section ---
            gr.Markdown("---\n#### 📱 Telegram Bot Configuration")
            gr.Markdown("Configure your Telegram bot token and allowed user ID.")

            current_token, current_user_id = _get_current_telegram_settings()

            telegram_token_input = gr.Textbox(
                label="Bot Token",
                value=current_token,
                type="password",
                placeholder="123456789:ABCdefGHIjklMNOpqrsTUVwxyz",
            )
            allowed_user_input = gr.Textbox(
                label="Allowed User ID",
                value=current_user_id,
                placeholder="Your Telegram user ID (numbers only)",
            )

            telegram_config_status = gr.Textbox(label="Status", interactive=False)
            save_telegram_config_btn = gr.Button("Save Telegram Settings", variant="primary")
            save_telegram_config_btn.click(
                _save_telegram_bot_settings,
                inputs=[telegram_token_input, allowed_user_input],
                outputs=[telegram_config_status],
            )

            # --- Personality Section ---
            gr.Markdown("---\n#### 💭 Personality Configuration")
            gr.Markdown("Edit the core personality (soul.md) and personas (persona.md).")

            soul_content = _load_soul_content()
            persona_content = _load_persona_content()

            soul_editor = gr.TextArea(
                label="soul.md - Core Personality",
                value=soul_content,
                lines=10,
                max_lines=20,
            )
            soul_save_status = gr.Textbox(label="Status", interactive=False)
            save_soul_btn = gr.Button("Save Soul", variant="secondary")
            save_soul_btn.click(
                _save_soul_content,
                inputs=[soul_editor],
                outputs=[soul_save_status],
            )

            persona_editor = gr.TextArea(
                label="persona.md - Persona Definitions",
                value=persona_content,
                lines=10,
                max_lines=30,
            )
            persona_save_status = gr.Textbox(label="Status", interactive=False)
            save_persona_btn = gr.Button("Save Personas", variant="secondary")
            save_persona_btn.click(
                _save_persona_content,
                inputs=[persona_editor],
                outputs=[persona_save_status],
            )

            # --- TTS Provider Selection ---
            gr.Markdown("---\n#### TTS Provider Selection")
            gr.Markdown("Choose between local (IndexTTS2) or cloud (ElevenLabs) TTS.")

            # Get current settings
            current_settings = _get_tts_settings()
            (current_provider, current_ref_audio, current_emotion, current_speed,
             current_quality, current_api_key, current_voice_name) = current_settings

            with gr.Row():
                with gr.Column():
                    tts_provider_dropdown = gr.Dropdown(
                        label="TTS Provider",
                        choices=TTS_PROVIDER_OPTIONS,
                        value=current_provider,
                    )
                    gr.Markdown("local = IndexTTS2 (free, GPU required) | elevenlabs = Cloud API (fast, quality)")

                with gr.Column():
                    elevenlabs_voice_dropdown = gr.Dropdown(
                        label="ElevenLabs Voice",
                        choices=ELEVENLABS_VOICE_OPTIONS,
                        value=current_voice_name,
                    )
                    gr.Markdown("Select voice for ElevenLabs")

            with gr.Row():
                elevenlabs_api_key_input = gr.Textbox(
                    label="ElevenLabs API Key",
                    value=current_api_key,
                    type="password",
                    placeholder="Enter your ElevenLabs API key..."
                )

            # --- Local TTS Settings (IndexTTS2) ---
            gr.Markdown("---\n#### Local TTS Settings (IndexTTS2)")
            gr.Markdown("Configure local voice synthesis (requires GPU).")

            # Get available reference audio files
            ref_audio_choices = get_reference_audio_list()
            if not ref_audio_choices:
                ref_audio_choices = ["No reference audio found"]

            with gr.Row():
                with gr.Column():
                    reference_audio_dropdown = gr.Dropdown(
                        label="Reference Audio (Voice Timbre)",
                        choices=ref_audio_choices,
                        value=current_ref_audio if current_ref_audio in ref_audio_choices else ref_audio_choices[0],
                    )
                    gr.Markdown("Select audio file to clone voice timbre")

                with gr.Column():
                    emotion_dropdown = gr.Dropdown(
                        label="Emotion",
                        choices=EMOTION_OPTIONS,
                        value=current_emotion,
                    )
                    gr.Markdown("Options: neutral, happy, sad, angry, excited, calm")

            with gr.Row():
                with gr.Column():
                    speed_slider = gr.Slider(
                        minimum=0.5,
                        maximum=2.0,
                        value=current_speed,
                        step=0.1,
                        label="Speed (0.5x - 2.0x)"
                    )

                with gr.Column():
                    quality_dropdown = gr.Dropdown(
                        label="Quality Preset",
                        choices=QUALITY_PRESET_OPTIONS,
                        value=current_quality,
                    )
                    gr.Markdown("fast=fastest, balanced=default, quality=best")

            tts_status = gr.Textbox(label="Status", interactive=False)
            save_tts_btn = gr.Button("Save TTS Settings", variant="primary")
            save_tts_btn.click(
                _save_tts_settings,
                inputs=[tts_provider_dropdown, reference_audio_dropdown, emotion_dropdown,
                        speed_slider, quality_dropdown, elevenlabs_api_key_input,
                        elevenlabs_voice_dropdown],
                outputs=[tts_status],
            )

            # --- Voice Preview Section ---
            gr.Markdown("---\n#### Voice Preview")
            gr.Markdown("Enter custom text to preview how the voice will sound.")

            with gr.Row():
                preview_text = gr.Textbox(
                    label="Preview Text",
                    placeholder="Enter text to synthesize...",
                    lines=2,
                )

            with gr.Row():
                preview_btn = gr.Button("Generate Preview", variant="secondary")

            with gr.Row():
                preview_audio = gr.Audio(label="Generated Speech", type="filepath")
                preview_status = gr.Textbox(label="Status", lines=1)

            preview_btn.click(
                preview_voice,
                inputs=[preview_text, reference_audio_dropdown, emotion_dropdown, speed_slider, quality_dropdown],
                outputs=[preview_audio, preview_status],
            )

    outputs = [brain_status, senses_status, bot_status, brain_log, senses_log, bot_log]
    demo.load(refresh, outputs=outputs)
    refresh_btn.click(refresh, outputs=outputs)
    start_brain_btn.click(start_brain, outputs=outputs)
    stop_brain_btn.click(stop_brain, outputs=outputs)
    start_senses_btn.click(start_senses, outputs=outputs)
    stop_senses_btn.click(stop_senses, outputs=outputs)
    start_bot_btn.click(start_bot, outputs=outputs)
    stop_bot_btn.click(stop_bot, outputs=outputs)
    start_all_btn.click(start_all, outputs=outputs)
    stop_all_btn.click(stop_all, outputs=outputs)

    # Page load: start with new conversation (empty chat) and load conversation list
    def init_chat_tab(profile_name):
        try:
            user_id = _profile_user_id(profile_name)
            migrate_old_chat_logs(user_id)
            new_id = create_conversation(user_id, "New Chat")
            convs = get_conversation_list(profile_name)
            choices = [f"{id} - {title} ({time})" for id, title, time in convs]
            # 重点：新建对话后，让下拉框选中新建的对话
            value = f"{new_id} - New Chat ({datetime.now().strftime('%Y-%m-%d %H:%M')})"
            # 但新建对话可能还没有更新时间，简单点直接用 choices 中匹配 new_id 的那一项
            # 更稳健：从 choices 中找出以 new_id 开头的项
            selected = next((c for c in choices if c.startswith(f"{new_id} -")), choices[0] if choices else None)
            return [], [], new_id, "New Chat", gr.update(choices=choices, value=selected)
        except Exception as e:
            print(f"Error initializing chat tab: {e}")
            return [], [], None, "New Chat", gr.update(choices=[], value=None)

    demo.load(init_chat_tab, inputs=[profile], outputs=[chat, state, conversation_id, conversation_title, conversation_dropdown])

    # Refresh conversation list - returns simple list of choices for dropdown
    def refresh_conversation_list(profile_name, current_conv_id=None):
        try:
            convs = get_conversation_list(profile_name)
            choices = [f"{id} - {title} ({time})" for id, title, time in convs]
            # 如果传入了当前对话ID，尝试选中它；否则选中列表第一个
            if current_conv_id:
                value = next((c for c in choices if c.startswith(f"{current_conv_id} -")), choices[0] if choices else None)
            else:
                value = choices[0] if choices else None
            return gr.update(choices=choices, value=value)
        except Exception as e:
            print(f"Error refreshing conversation list: {e}")
            return gr.update(choices=[], value=None)

    # Load selected conversation
    def on_load_conversation(profile_name, selected_conv_id):
        try:
            if selected_conv_id is None:
                return [], None, "Select a conversation to load", []
            if isinstance(selected_conv_id, list):
                if len(selected_conv_id) == 0:
                    return [], None, "Select a conversation to load", []
            selected_conv_id = selected_conv_id[0]
            try:
               conv_id = int(selected_conv_id.split(" - ")[0])
            except (ValueError, IndexError, AttributeError):
                return [], None, "Invalid conversation selection", []
            history, _, title = load_conversation(profile_name, conv_id)
        # 返回 chat 和 state 相同的历史记录
            return history, conv_id, title, history
        except Exception as e:
            print(f"Error loading conversation: {e}")
            return [], None, f"Error: {str(e)}", []

    # Create new conversation
    def on_new_conversation(profile_name):
        try:
            history, conv_id, title = start_new_conversation(profile_name)
            convs = get_conversation_list(profile_name)
            choices = [f"{id} - {title} ({time})" for id, title, time in convs]
            value = next((c for c in choices if c.startswith(f"{conv_id} -")), choices[0] if choices else None)
            return [], [], conv_id, title, gr.update(choices=choices, value=value)
        except Exception as e:
            print(f"Error creating new conversation: {e}")
            return [], None, "New Chat", gr.update(choices=[], value=None)

    # Delete conversation
    def on_delete_conversation(profile_name, selected_conv_id, current_conv_id):
        try:
            if selected_conv_id is None:
                return "Select a conversation to delete", gr.update(choices=[], value=None), None, "New Chat", [], []
            if isinstance(selected_conv_id, list):
                if len(selected_conv_id) == 0:
                    return "Select a conversation to delete", gr.update(choices=[], value=None), None, "New Chat", [], []
                selected_conv_id = selected_conv_id[0]
            try:
                conv_id = int(selected_conv_id.split(" - ")[0])
            except (ValueError, IndexError, AttributeError):
                return "Invalid conversation selection", gr.update(choices=[], value=None), None, "New Chat", [], []

            msg = remove_conversation(profile_name, conv_id)

            if msg.startswith("Delete failed"):
            # 删除失败：重新获取列表，保持当前对话不变
                convs = get_conversation_list(profile_name)
                choices = [f"{id} - {title} ({time})" for id, title, time in convs]
                value = next((c for c in choices if c.startswith(f"{current_conv_id} -")), choices[0] if choices else None)
            # 重新加载当前对话的历史
                history, _, title = load_conversation(profile_name, current_conv_id)
                return msg, gr.update(choices=choices, value=value), current_conv_id, title, history, history

        # 删除成功，获取最新列表
            convs = get_conversation_list(profile_name)
            choices = [f"{id} - {title} ({time})" for id, title, time in convs]

            if current_conv_id == conv_id:
            # 删除的是当前对话：清空聊天区域，不选中任何对话
                return msg, gr.update(choices=choices, value=None), None, "New Chat", [], []
            else:
            # 删除的是其他对话：保持当前对话不变
                history, _, title = load_conversation(profile_name, current_conv_id)
                value = next((c for c in choices if c.startswith(f"{current_conv_id} -")), choices[0] if choices else None)
                return msg, gr.update(choices=choices, value=value), current_conv_id, title, history, history
        except Exception as e:
            print(f"Error deleting conversation: {e}")
            return f"Error: {str(e)}", gr.update(choices=[], value=None), None, "New Chat", [], []

    # Button event bindings
    # Refresh conversation list on page load and when button clicked
    refresh_conv_btn.click(
        refresh_conversation_list,
        inputs=[profile, conversation_id],
        outputs=[conversation_dropdown],
    )

    load_conv_btn.click(
        on_load_conversation,
        inputs=[profile, conversation_dropdown],
        outputs=[chat, conversation_id, conversation_title, state],
    )

    new_conv_btn.click(
        on_new_conversation,
        inputs=[profile],
        outputs=[chat, state, conversation_id, conversation_title, conversation_dropdown],
    )

    delete_conv_btn.click(
        on_delete_conversation,
        inputs=[profile, conversation_dropdown, conversation_id],
        outputs=[status, conversation_dropdown, conversation_id, conversation_title, chat, state],
    )

    send_btn.click(
        send_text,
        inputs=[profile, chat_in, state, voice_toggle, web_search_toggle, conversation_id],
        outputs=[chat, status, chat_in, voice_out, state, conversation_id],
    )

    send_audio_btn.click(
        send_uploaded_audio,
        inputs=[profile, audio_in, state, voice_toggle, conversation_id],
        outputs=[chat, status, chat_in, voice_out, state, conversation_id],
    )

    clear_btn.click(clear_chat, outputs=[chat, status, chat_in, voice_out, state])

    call_on.change(toggle_call, inputs=[call_on], outputs=[call_state, call_status])
    calibrate_btn.click(calibrate_noise, inputs=[noise_clip], outputs=[noise_threshold, call_status])
    process_turn_btn.click(
        process_call_turn,
        inputs=[call_profile, call_on, noise_threshold, call_audio_in, call_chat_state],
        outputs=[call_chat, call_state, call_status, call_voice_out, call_chat_state],  # 添加call_chat_state到outputs以持久化历史记录
    )

    # Telegram Bot tab handlers
    save_telegram_btn.click(
        _save_telegram_settings,
        inputs=[telegram_voice_dropdown, telegram_mode_radio],
        outputs=[telegram_status, current_settings_display],
    )


demo.launch(server_name="127.0.0.1", server_port=7860)