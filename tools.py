from __future__ import annotations

import json
import httpx
from pathlib import Path


VOICE_CONFIG_PATH = Path(__file__).resolve().parent / "voice_config.json"


def get_voice_config() -> dict:
    """Get voice settings from voice_config.json"""
    default_config = {
        "voice_enabled": False,
        "voice_name": "Pebble",
        "reference_audio": None,
        "emotion": "neutral",
        "speed": 1.0,
        "quality_preset": "balanced",
        "tts_provider": "local",
        "elevenlabs_api_key": "",
        "elevenlabs_voice_id": "21m00Tcm4TlvDq8ikWAM"
    }
    try:
        if VOICE_CONFIG_PATH.exists():
            config = json.loads(VOICE_CONFIG_PATH.read_text())
            return {**default_config, **config}
    except Exception:
        pass
    return default_config


def set_voice_config(
    voice_enabled: bool = None,
    voice_name: str = None,
    reference_audio: str = None,
    emotion: str = None,
    speed: float = None,
    quality_preset: str = None,
    tts_provider: str = None,
    elevenlabs_api_key: str = None,
    elevenlabs_voice_id: str = None
) -> None:
    """Update voice settings in voice_config.json"""
    config = get_voice_config()
    if voice_enabled is not None:
        config["voice_enabled"] = voice_enabled
    if voice_name is not None:
        config["voice_name"] = voice_name
    if reference_audio is not None:
        config["reference_audio"] = reference_audio
    if emotion is not None:
        config["emotion"] = emotion
    if speed is not None:
        config["speed"] = speed
    if quality_preset is not None:
        config["quality_preset"] = quality_preset
    if tts_provider is not None:
        config["tts_provider"] = tts_provider
    if elevenlabs_api_key is not None:
        config["elevenlabs_api_key"] = elevenlabs_api_key
    if elevenlabs_voice_id is not None:
        config["elevenlabs_voice_id"] = elevenlabs_voice_id
    VOICE_CONFIG_PATH.write_text(json.dumps(config, indent=2))


def get_current_weather(city: str) -> str:
    """Get current weather using wttr.in API."""
    if not city.strip():
        return "unknown weather"

    safe_city = city.strip().replace(" ", "+")
    url = f"https://wttr.in/{safe_city}?format=j1"

    try:
        # Try httpx first
        response = httpx.get(url, timeout=5.0)
        response.raise_for_status()
        payload = response.json()
    except Exception:
        # Fallback to urllib if httpx fails
        try:
            import urllib.request
            import ssl
            # Create unverified SSL context for wttr.in
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=10, context=ctx) as resp:
                import json
                payload = json.load(resp)
        except Exception as e:
            print(f"[Weather Error] Failed to fetch weather: {e}")
            return "unknown weather"

    try:
        current = payload.get("current_condition", [{}])[0]
        temp_c = current.get("temp_C")
        weather_desc_arr = current.get("weatherDesc", [])
        condition = weather_desc_arr[0].get("value") if weather_desc_arr else None

        temp_text = f"{temp_c}°C" if temp_c is not None else "Unknown"
        condition_text = condition or "Unknown"
        return f"{condition_text}, {temp_text}"
    except Exception as e:
        print(f"[Weather Parse Error] {e}")
        return "unknown weather"


# Alias for backwards compatibility
def get_weather(city: str) -> str:
    return get_current_weather(city)


# Reference audio directory for IndexTTS2
def get_reference_audio_list() -> list:
    """Get list of available reference audio files from index-tts/examples"""
    examples_dir = Path(__file__).resolve().parent / "index-tts" / "examples"
    if not examples_dir.exists():
        return []
    wav_files = list(examples_dir.glob("*.wav"))
    return [f.name for f in wav_files]


def get_reference_audio_path(filename: str) -> str:
    """Get full path to reference audio file"""
    examples_dir = Path(__file__).resolve().parent / "index-tts" / "examples"
    return str(examples_dir / filename)


# ==================== Web Search Functions ====================

def web_search(query: str, max_results: int = 5) -> str:
    """
    Search the web using DuckDuckGo.

    Args:
        query: Search query string
        max_results: Maximum number of results to return (default: 5)

    Returns:
        Formatted string with search results, or empty string if failed
    """
    if not query or not query.strip():
        return ""

    query = query.strip()

    # Try to use duckduckgo package first
    try:
        from duckduckgo import DDGS
        ddgs = DDGS()
        results = list(ddgs.text(query, max_results=max_results))

        if not results:
            return ""

        formatted_results = []
        for i, result in enumerate(results, 1):
            title = result.get('title', 'No title')
            body = result.get('body', 'No description')
            href = result.get('href', '')

            # Limit body text length
            if len(body) > 200:
                body = body[:200] + "..."

            formatted_results.append(f"{i}. {title}\n   {body}")

        return "\n\n".join(formatted_results)

    except ImportError:
        # Fallback: use direct HTTP request to DuckDuckGo HTML
        return _web_search_fallback(query, max_results)
    except Exception as e:
        print(f"[Web Search Error] duckduckgo package failed: {e}")
        # Try fallback
        try:
            return _web_search_fallback(query, max_results)
        except Exception:
            return ""


def _web_search_fallback(query: str, max_results: int = 5) -> str:
    """
    Fallback web search using DuckDuckGo HTML (no API key needed).
    """
    import re

    # Use DuckDuckGo HTML search
    url = f"https://html.duckduckgo.com/html/?q={query.replace(' ', '+')}"

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    try:
        response = httpx.get(url, headers=headers, timeout=10.0)
        response.raise_for_status()
        html = response.text

        # Parse results from HTML
        results = []

        # Match result blocks
        result_pattern = r'<a rel="nofollow" class="result__a" href="[^"]*q=([^&"]+)'
        title_pattern = r'<a rel="nofollow" class="result__a"[^>]*>(.+?)</a>'
        snippet_pattern = r'<a class="result__snippet"[^>]*>(.+?)</a>'

        # Simple HTML parsing
        import re

        # Find all result blocks
        blocks = re.findall(r'<div class="result__body">(.*?)</div>', html, re.DOTALL)

        for block in blocks[:max_results]:
            # Extract title
            title_match = re.search(r'<a[^>]*class="result__a"[^>]*>(.+?)</a>', block, re.DOTALL)
            title = title_match.group(1) if title_match else "No title"

            # Clean HTML tags from title
            title = re.sub(r'<[^>]+>', '', title).strip()

            # Extract snippet
            snippet_match = re.search(r'<a[^>]*class="result__snippet"[^>]*>(.+?)</a>', block, re.DOTALL)
            snippet = snippet_match.group(1) if snippet_match else ""

            # Clean HTML tags from snippet
            snippet = re.sub(r'<[^>]+>', '', snippet).strip()

            # Limit snippet length
            if len(snippet) > 200:
                snippet = snippet[:200] + "..."

            if title:
                results.append(f"{len(results) + 1}. {title}\n   {snippet}")

        if results:
            return "\n\n".join(results)
        else:
            return ""

    except Exception as e:
        print(f"[Web Search Fallback Error] {e}")
        return ""


def needs_web_search(query: str) -> bool:
    """
    Determine if a query needs web search.

    Args:
        query: User's input query

    Returns:
        True if search is likely needed, False otherwise
    """
    if not query:
        return False

    query_lower = query.lower()

    # Keywords that typically require web search
    search_triggers = [
        # Time/date - ADDED MORE
        "what time", "current time", "what's the time", "time is it",
        "today's date", "what day is it", "what year",
        # Weather (beyond user's location)
        "weather in", "temperature in", "forecast in",
        # News/events
        "news", "latest", "recent", "what happened", "breaking",
        # Information queries
        "what is", "who is", "where is", "how to", "why does",
        "definition", "meaning of", "capital of", "population of",
        # Current events
        "election", "winner", "result", "score", "game",
        # Real-time info
        "price of", "cost of", "stock", "bitcoin", "exchange rate",
        # Comparisons
        "compare", "versus", "vs",
        # How to do something
        "how do i", "how can i", "steps to", "tutorial",
    ]

    # Check if any trigger keyword is in the query
    for trigger in search_triggers:
        if trigger in query_lower:
            return True

    return False


def extract_search_query(query: str) -> str:
    """
    Extract a clean search query from user's input.

    Args:
        query: User's input query

    Returns:
        Cleaned search query string
    """
    import re

    # Remove common prefixes
    query = query.strip()

    # Common patterns to remove (bot names, greetings, etc.)
    prefixes_to_remove = [
        "search for",
        "look up",
        "find",
        "google",
        "what is",
        "who is",
        "tell me about",
        "can you",
        "please",
        # Bot names and greetings
        r"^hey\s+pebble\.?\s*",
        r"^pebble\s+",
        r"^hey\s+",
        r"^yo\s+",
        r"^hi\s+",
        r"^hello\s+",
        # Common English response prefixes (these get included in history)
        r"^yes[,\s]+",
        r"^yeah[,\s]+",
        r"^sure[,\s]+",
        r"^ok[,\s]+",
        r"^okay[,\s]+",
        r"^so[,\s]+",
        r"^well[,\s]+",
        r"^actually[,\s]+",
        r"^basically[,\s]+",
        r"^honestly[,\s]+",
        r"^right[,\s]+",
        r"^alright[,\s]+",
    ]

    query_lower = query.lower()

    # Use regex for more complex patterns
    for prefix in prefixes_to_remove:
        if isinstance(prefix, str):
            # Simple string prefix
            if query_lower.startswith(prefix):
                query = query[len(prefix):].strip()
                query_lower = query.lower()
        else:
            # Regex pattern
            query = re.sub(prefix, '', query_lower, flags=re.IGNORECASE).strip()
            query_lower = query.lower()

    # Also clean up if query contains multiple questions (take the first one)
    # e.g., "what time is it? What is Bitcoin?" -> "what time is it?"
    if '?' in query:
        query = query.split('?')[0].strip() + "?"

    return query
# tools.py 末尾添加

VISION_CONFIG_PATH = Path(__file__).resolve().parent / "vision_config.json"

def get_vision_config() -> dict:
    """Get vision model settings from vision_config.json"""
    default_config = {
        "enabled": False,
        "provider": "阿里千问",
        "api_key": "",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-vl-plus",
    }
    try:
        if VISION_CONFIG_PATH.exists():
            config = json.loads(VISION_CONFIG_PATH.read_text())
            return {**default_config, **config}
    except Exception:
        pass
    return default_config

def set_vision_config(
    enabled: bool = None,
    provider: str = None,
    api_key: str = None,
    base_url: str = None,
    model: str = None,
) -> None:
    """Update vision settings in vision_config.json"""
    config = get_vision_config()
    if enabled is not None:
        config["enabled"] = enabled
    if provider is not None:
        config["provider"] = provider
    if api_key is not None:
        config["api_key"] = api_key
    if base_url is not None:
        config["base_url"] = base_url
    if model is not None:
        config["model"] = model
VISION_CONFIG_PATH = Path(__file__).resolve().parent / "vision_config.json"
def get_vision_config() -> dict:
    """Get vision model settings from vision_config.json"""
    default_config = {
        "enabled": False,
        "provider": "aliyun",
        "api_key": "",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-vl-plus",
    }
    try:
        if VISION_CONFIG_PATH.exists():
            config = json.loads(VISION_CONFIG_PATH.read_text())
            return {**default_config, **config}
    except Exception:
        pass
    return default_config
def set_vision_config(
    enabled: bool = None,
    provider: str = None,
    api_key: str = None,
    base_url: str = None,
    model: str = None,
) -> None:
    """Update vision settings in vision_config.json"""
    config = get_vision_config()
    if enabled is not None:
        config["enabled"] = enabled
    if provider is not None:
        config["provider"] = provider
    if api_key is not None:
        config["api_key"] = api_key
    if base_url is not None:
        config["base_url"] = base_url
    if model is not None:
        config["model"] = model
    VISION_CONFIG_PATH.write_text(json.dumps(config, indent=2))
