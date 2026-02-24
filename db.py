import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "brook.db"
VOICE_CONFIG_PATH = BASE_DIR / "true_voices.json"


from prompts import load_persona_prompt

# Load persona prompts from persona.md
PERSONA_SEEDS = {
    "Fun Pebble": load_persona_prompt("Fun Pebble"),
    "Executive Pebble": load_persona_prompt("Executive Pebble"),
    "Fitness Pebble": load_persona_prompt("Fitness Pebble"),
}


def get_connection() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS personas (
                mode TEXT PRIMARY KEY,
                system_prompt TEXT NOT NULL,
                is_custom INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS active_context (
                user_id TEXT PRIMARY KEY,
                current_mode TEXT NOT NULL,
                custom_persona_description TEXT,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (current_mode) REFERENCES personas(mode)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS user_profiles (
                user_id TEXT PRIMARY KEY,
                bot_name TEXT,
                user_name TEXT,
                summary TEXT NOT NULL DEFAULT '',
                emotional_notes TEXT NOT NULL DEFAULT '',
                day_summary TEXT NOT NULL DEFAULT '',
                location TEXT NOT NULL DEFAULT '',
                relationship_status TEXT NOT NULL DEFAULT 'We are getting to know each other.',
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS reminders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                reminder_text TEXT NOT NULL,
                trigger_time TEXT NOT NULL,
                job_id TEXT UNIQUE,
                status TEXT NOT NULL DEFAULT 'scheduled',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS voice_settings (
                user_id TEXT PRIMARY KEY,
                active_voice_name TEXT NOT NULL DEFAULT 'Pebble',
                voice_mode TEXT NOT NULL DEFAULT 'off',
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(user_profiles)").fetchall()
        }
        if "location" not in columns:
            conn.execute(
                "ALTER TABLE user_profiles ADD COLUMN location TEXT NOT NULL DEFAULT ''"
            )
        if "relationship_status" not in columns:
            conn.execute(
                "ALTER TABLE user_profiles ADD COLUMN relationship_status TEXT NOT NULL DEFAULT 'We are getting to know each other.'"
            )
        conn.commit()

    seed_personas()


def seed_personas() -> None:
    with get_connection() as conn:
        conn.execute("DELETE FROM personas")

        for mode, system_prompt in PERSONA_SEEDS.items():
            conn.execute(
                """
                INSERT INTO personas (mode, system_prompt, is_custom)
                VALUES (?, ?, 0)
                """,
                (mode, system_prompt),
            )

        conn.execute(
            """
            INSERT INTO personas (mode, system_prompt, is_custom)
            VALUES ('Custom', 'Custom persona generated from user description.', 1)
            """
        )

        # Backward compatibility: migrate users on legacy mode names.
        conn.execute(
            """
            UPDATE active_context
            SET current_mode = CASE current_mode
                WHEN 'Girlfriend (Nani)' THEN 'Fun Pebble'
                WHEN 'Executive' THEN 'Executive Pebble'
                WHEN 'Health Coach' THEN 'Fitness Pebble'
                ELSE current_mode
            END
            """
        )
        conn.commit()


def get_personas() -> List[Dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT mode, system_prompt, is_custom FROM personas ORDER BY is_custom, mode"
        ).fetchall()
    return [dict(row) for row in rows]


def get_persona_by_mode(mode: str) -> Optional[Dict[str, Any]]:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT mode, system_prompt, is_custom FROM personas WHERE mode = ?",
            (mode,),
        ).fetchone()
    return dict(row) if row else None


def update_persona_prompt(mode: str, system_prompt: str) -> None:
    now = datetime.utcnow().isoformat()
    with get_connection() as conn:
        conn.execute(
            """
            UPDATE personas
            SET system_prompt = ?, updated_at = ?
            WHERE mode = ?
            """,
            (system_prompt, now, mode),
        )
        conn.commit()


def set_active_mode(user_id: str, mode: str, custom_description: Optional[str] = None) -> None:
    now = datetime.utcnow().isoformat()
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO active_context (user_id, current_mode, custom_persona_description, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                current_mode = excluded.current_mode,
                custom_persona_description = excluded.custom_persona_description,
                updated_at = excluded.updated_at
            """,
            (user_id, mode, custom_description, now),
        )
        conn.commit()


def get_active_mode(user_id: str, default_mode: str = "Fun Pebble") -> str:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT current_mode FROM active_context WHERE user_id = ?", (user_id,)
        ).fetchone()
    return row["current_mode"] if row else default_mode


def _default_voice_name() -> str:
    try:
        if VOICE_CONFIG_PATH.exists():
            payload = json.loads(VOICE_CONFIG_PATH.read_text())
            if isinstance(payload, list) and payload:
                name = str(payload[0].get("name", "")).strip()
                if name:
                    return name
    except Exception:
        pass
    return "Pebble"  # Valid voice from true_voices.json


def get_voice_settings(user_id: str) -> Dict[str, str]:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT active_voice_name, voice_mode FROM voice_settings WHERE user_id = ?",
            (user_id,),
        ).fetchone()

    if not row:
        return {
            "active_voice_name": _default_voice_name(),
            "voice_mode": "off",  # Default: text only
        }
    return dict(row)


def upsert_voice_settings(
    user_id: str,
    active_voice_name: Optional[str] = None,
    voice_mode: Optional[str] = None,
) -> None:
    now = datetime.utcnow().isoformat()
    existing = get_voice_settings(user_id)
    active_voice_value = active_voice_name or existing.get("active_voice_name", _default_voice_name())
    voice_mode_value = voice_mode or existing.get("voice_mode", "off")  # Default to text only

    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO voice_settings (user_id, active_voice_name, voice_mode, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                active_voice_name = excluded.active_voice_name,
                voice_mode = excluded.voice_mode,
                updated_at = excluded.updated_at
            """,
            (user_id, active_voice_value, voice_mode_value, now),
        )
        conn.commit()


def update_voice_setting(user_id: str, key: str, value: str) -> None:
    if key not in {"active_voice_name", "voice_mode"}:
        raise ValueError(f"Unsupported voice setting key: {key}")

    existing = get_voice_settings(user_id)
    upsert_voice_settings(
        user_id=user_id,
        active_voice_name=value if key == "active_voice_name" else existing.get("active_voice_name"),
        voice_mode=value if key == "voice_mode" else existing.get("voice_mode"),
    )


def log_chat(user_id: str, role: str, content: str) -> None:
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO chat_logs (user_id, role, content) VALUES (?, ?, ?)",
            (user_id, role, content),
        )
        conn.commit()


def get_recent_chat_logs(user_id: str, limit: int = 50) -> List[Dict[str, str]]:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT role, content, created_at
            FROM chat_logs
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (user_id, limit),
        ).fetchall()

    results = [dict(row) for row in rows]
    results.reverse()
    return results


def get_chat_logs_for_day(user_id: str, day_iso: str) -> List[Dict[str, str]]:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT role, content, created_at
            FROM chat_logs
            WHERE user_id = ?
              AND DATE(created_at) = DATE(?)
            ORDER BY id ASC
            """,
            (user_id, day_iso),
        ).fetchall()

    return [dict(row) for row in rows]


def list_users_with_logs() -> List[str]:
    with get_connection() as conn:
        rows = conn.execute("SELECT DISTINCT user_id FROM chat_logs").fetchall()
    return [row["user_id"] for row in rows]


def get_user_profile(user_id: str) -> Dict[str, str]:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT bot_name, user_name, summary, emotional_notes, day_summary, location, relationship_status FROM user_profiles WHERE user_id = ?",
            (user_id,),
        ).fetchone()

    if not row:
        return {
            "bot_name": "",
            "user_name": "",
            "summary": "",
            "emotional_notes": "",
            "day_summary": "",
            "location": "",
            "relationship_status": "We are getting to know each other.",
        }
    return dict(row)


def upsert_user_profile(
    user_id: str,
    summary: str = "",
    emotional_notes: str = "",
    day_summary: str = "",
    location: Optional[str] = None,
    relationship_status: Optional[str] = None,
    bot_name: Optional[str] = None,
    user_name: Optional[str] = None,
) -> None:
    now = datetime.utcnow().isoformat()
    existing = get_user_profile(user_id)
    location_value = location if location is not None else existing.get("location", "")
    relationship_value = (
        relationship_status
        if relationship_status is not None
        else existing.get("relationship_status", "We are getting to know each other.")
    )
    bot_name_value = bot_name if bot_name is not None else existing.get("bot_name", "")
    user_name_value = user_name if user_name is not None else existing.get("user_name", "")
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO user_profiles (user_id, bot_name, user_name, summary, emotional_notes, day_summary, location, relationship_status, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                bot_name = COALESCE(excluded.bot_name, bot_name),
                user_name = COALESCE(excluded.user_name, user_name),
                summary = excluded.summary,
                emotional_notes = excluded.emotional_notes,
                day_summary = excluded.day_summary,
                location = excluded.location,
                relationship_status = excluded.relationship_status,
                updated_at = excluded.updated_at
            """,
            (
                user_id,
                bot_name_value,
                user_name_value,
                summary,
                emotional_notes,
                day_summary,
                location_value,
                relationship_value,
                now,
            ),
        )
        conn.commit()


def update_user_location(user_id: str, location: str) -> None:
    """Update user's location for weather tracking."""
    upsert_user_profile(user_id=user_id, location=location)


# ==================== Conversation Management ====================

def init_conversations_table() -> None:
    """Initialize conversation management table"""
    with get_connection() as conn:
        # Create conversations table
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                title TEXT NOT NULL DEFAULT 'New Chat',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        # Check if chat_logs table has conversation_id field, add if not
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(chat_logs)").fetchall()}
        if "conversation_id" not in columns:
            conn.execute(
                "ALTER TABLE chat_logs ADD COLUMN conversation_id INTEGER"
            )

        conn.commit()


def create_conversation(user_id: str, title: str = "New Chat") -> int:
    """Create new conversation, returns conversation ID"""
    now = datetime.utcnow().isoformat()
    with get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO conversations (user_id, title, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            """,
            (user_id, title, now, now),
        )
        conn.commit()
        return cursor.lastrowid


def get_conversations(user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
    """获取用户的所有会话列表"""
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, title, created_at, updated_at
            FROM conversations
            WHERE user_id = ?
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (user_id, limit),
        ).fetchall()
    return [dict(row) for row in rows]


def get_conversation(conversation_id: int) -> Optional[Dict[str, Any]]:
    """获取单个会话信息"""
    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT id, user_id, title, created_at, updated_at
            FROM conversations
            WHERE id = ?
            """,
            (conversation_id,),
        ).fetchone()
    return dict(row) if row else None


def update_conversation_title(conversation_id: int, title: str) -> None:
    """更新会话标题"""
    now = datetime.utcnow().isoformat()
    with get_connection() as conn:
        conn.execute(
            """
            UPDATE conversations
            SET title = ?, updated_at = ?
            WHERE id = ?
            """,
            (title, now, conversation_id),
        )
        conn.commit()


def update_conversation_time(conversation_id: int) -> None:
    """更新会话的最后活动时间"""
    now = datetime.utcnow().isoformat()
    with get_connection() as conn:
        conn.execute(
            """
            UPDATE conversations
            SET updated_at = ?
            WHERE id = ?
            """,
            (now, conversation_id),
        )
        conn.commit()


def delete_conversation(conversation_id: int) -> None:
    """Delete conversation (and associated chat logs)"""
    with get_connection() as conn:
        # First delete associated chat logs
        conn.execute(
            "DELETE FROM chat_logs WHERE conversation_id = ?",
            (conversation_id,),
        )
        # Then delete the conversation
        conn.execute(
            "DELETE FROM conversations WHERE id = ?",
            (conversation_id,),
        )
        conn.commit()


def add_chat_log_with_conversation(conversation_id: int, user_id: str, role: str, content: str) -> None:
    """Save chat log with conversation ID"""
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO chat_logs (conversation_id, user_id, role, content) VALUES (?, ?, ?, ?)",
            (conversation_id, user_id, role, content),
        )
        # Update conversation's last update time
        conn.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (datetime.utcnow().isoformat(), conversation_id),
        )
        conn.commit()


def get_chat_logs_by_conversation(conversation_id: int) -> List[Dict[str, str]]:
    """Get all chat logs for a specific conversation"""
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT role, content, created_at
            FROM chat_logs
            WHERE conversation_id = ?
            ORDER BY id ASC
            """,
            (conversation_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def migrate_old_chat_logs(user_id: str) -> int:
    """Migrate old chat logs (without conversation_id) to a new conversation.
    Returns the conversation_id created or existing conversation_id if already migrated."""
    with get_connection() as conn:
        # Check if there are any chat logs without conversation_id
        rows = conn.execute(
            """
            SELECT id FROM chat_logs
            WHERE user_id = ? AND conversation_id IS NULL
            ORDER BY id ASC
            LIMIT 1
            """,
            (user_id,),
        ).fetchall()

        if not rows:
            # Already migrated - get existing conversation
            conv = conn.execute(
                """
                SELECT id FROM conversations
                WHERE user_id = ?
                ORDER BY id DESC
                LIMIT 1
                """,
                (user_id,),
            ).fetchone()
            return conv["id"] if conv else None

        # Create a new conversation for old logs
        now = datetime.utcnow().isoformat()
        cursor = conn.execute(
            """
            INSERT INTO conversations (user_id, title, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            """,
            (user_id, "Old Chat", now, now),
        )
        conv_id = cursor.lastrowid

        # Update all old chat logs with the new conversation_id
        conn.execute(
            """
            UPDATE chat_logs
            SET conversation_id = ?
            WHERE user_id = ? AND conversation_id IS NULL
            """,
            (conv_id, user_id),
        )
        conn.commit()
        return conv_id


# Initialize conversation table on load
init_conversations_table()


if __name__ == "__main__":
    init_db()
    print(f"Database initialized at: {DB_PATH}")
