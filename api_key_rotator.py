"""
Rotador de API Keys de Google — round-robin con cooldown ante errores 403/429.

Módulo standalone (sin dependencias de proyecto) que puede ser importado
desde cualquier agente o servicio.

Uso:
    from api_key_rotator import google_key_rotator, invoke_with_retry, create_google_llm

    key = google_key_rotator.get_key()
    google_key_rotator.report_failure(key)

    llm = create_google_llm()
    response = invoke_with_retry(llm, messages)
"""

import os
import time
import threading
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class GoogleApiKeyRotator:
    """Round-robin rotator para múltiples Google API Keys.

    Lee las keys de la variable de entorno GOOGLE_API_KEYS (separadas por comas).
    Si no existe, hace fallback a GOOGLE_API_KEY (una sola key, sin rotación).
    """

    COOLDOWN_SECONDS = 60

    def __init__(self, env_var: str = "GOOGLE_API_KEYS", fallback_var: str = "GOOGLE_API_KEY"):
        self._lock = threading.Lock()
        self._keys: list[str] = []
        self._index: int = 0
        self._cooldowns: dict[str, float] = {}
        self._env_var = env_var
        self._fallback_var = fallback_var
        self._loaded = False
        
        # Intentar carga inicial (puede estar vacía si no se llamó a load_dotenv)
        self.load_keys()

    def load_keys(self):
        """Carga las API keys desde las variables de entorno."""
        raw = os.getenv(self._env_var, "")
        if raw:
            self._keys = [k.strip() for k in raw.split(",") if k.strip()]

        if not self._keys:
            fallback = os.getenv(self._fallback_var, "")
            if fallback:
                self._keys = [fallback.strip()]

        if self._keys:
            self._loaded = True
            print(
                f"🔑 GoogleApiKeyRotator: {len(self._keys)} key(s) cargadas "
                f"[{', '.join(f'...{k[-4:]}' for k in self._keys)}]"
            )
        else:
            if self._loaded:
                print("⚠️ GoogleApiKeyRotator: Las keys cargadas anteriormente se han perdido o vaciado.")

    @property
    def total_keys(self) -> int:
        if not self._keys:
            self.load_keys()
        return len(self._keys)

    def get_key(self) -> str:
        """Devuelve la siguiente key disponible (round-robin, saltea cooldowns)."""
        if not self._keys:
            self.load_keys()
        if not self._keys:
            return ""

        with self._lock:
            now = time.time()
            for _ in range(len(self._keys)):
                key = self._keys[self._index]
                self._index = (self._index + 1) % len(self._keys)

                cooldown_ts = self._cooldowns.get(key, 0)
                if now - cooldown_ts >= self.COOLDOWN_SECONDS:
                    self._cooldowns.pop(key, None)
                    print(f"🔑 Usando Google API Key ...{key[-4:]}")
                    return key

            oldest_key = min(self._keys, key=lambda k: self._cooldowns.get(k, 0))
            self._cooldowns.pop(oldest_key, None)
            print(f"⚠️ Todas las keys en cooldown. Forzando uso de ...{oldest_key[-4:]}")
            return oldest_key

    def report_failure(self, key: str):
        """Marca una key como fallida (cooldown de COOLDOWN_SECONDS)."""
        if key not in self._keys:
            return
        with self._lock:
            self._cooldowns[key] = time.time()
            print(f"🚫 Key ...{key[-4:]} en cooldown por {self.COOLDOWN_SECONDS}s")

    def clear_cooldowns(self):
        with self._lock:
            self._cooldowns.clear()


# ── Singleton ──────────────────────────────────────────────────────
google_key_rotator = GoogleApiKeyRotator()


def create_google_llm(
    model: str = "gemini-2.5-flash",
    temperature: float = 0.0,
    max_output_tokens: int = 8192,
    rotator: Optional[GoogleApiKeyRotator] = None,
):
    """Crea un ChatGoogleGenerativeAI con la siguiente key disponible."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    if rotator is None:
        rotator = google_key_rotator
    key = rotator.get_key()
    return ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        google_api_key=key,
    )


# ── Detección de errores de cuota ──────────────────────────────────
def _is_quota_error(exc: Exception) -> bool:
    err_str = str(exc).lower()
    return any(
        ind in err_str
        for ind in [
            "403", "429", "resource_exhausted", "resourceexhausted",
            "rate limit", "rate_limit", "quota", "too many requests",
        ]
    )


def _rebuild_llm(llm, new_key):
    """Re-crea un ChatGoogleGenerativeAI con una nueva key."""
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(
        model=getattr(llm, "model_name", None) or getattr(llm, "model", "gemini-2.5-flash"),
        temperature=getattr(llm, "temperature", 0.0),
        max_output_tokens=getattr(llm, "max_output_tokens", 8192),
        google_api_key=new_key,
    )


def invoke_with_retry(llm, messages, *, rotator=None, max_retries=0, base_wait=2.0):
    """Invoca LLM con retry automático ante 403/429, rotando la API key."""
    if rotator is None:
        rotator = google_key_rotator
    if max_retries == 0:
        max_retries = max(rotator.total_keys, 1)

    last_exc = None
    current_key = getattr(llm, "google_api_key", "") or ""

    for attempt in range(max_retries + 1):
        try:
            return llm.invoke(messages)
        except Exception as exc:
            last_exc = exc
            if not _is_quota_error(exc):
                raise
            print(f"⚠️ [Retry {attempt+1}/{max_retries}] Cuota excedida con key ...{current_key[-4:] if current_key else '????'}: {str(exc)[:200]}")
            if attempt >= max_retries:
                break
            if current_key:
                rotator.report_failure(current_key)
            new_key = rotator.get_key()
            llm = _rebuild_llm(llm, new_key)
            current_key = new_key
            wait = base_wait * (2 ** attempt)
            print(f"⏳ Esperando {wait:.1f}s antes de reintentar...")
            time.sleep(wait)

    raise last_exc


async def ainvoke_with_retry(llm, messages, *, rotator=None, max_retries=0, base_wait=2.0):
    """Versión async de invoke_with_retry."""
    import asyncio
    if rotator is None:
        rotator = google_key_rotator
    if max_retries == 0:
        max_retries = max(rotator.total_keys, 1)

    last_exc = None
    current_key = getattr(llm, "google_api_key", "") or ""

    for attempt in range(max_retries + 1):
        try:
            return await llm.ainvoke(messages)
        except Exception as exc:
            last_exc = exc
            if not _is_quota_error(exc):
                raise
            print(f"⚠️ [Async Retry {attempt+1}/{max_retries}] Cuota excedida con key ...{current_key[-4:] if current_key else '????'}: {str(exc)[:200]}")
            if attempt >= max_retries:
                break
            if current_key:
                rotator.report_failure(current_key)
            new_key = rotator.get_key()
            llm = _rebuild_llm(llm, new_key)
            current_key = new_key
            wait = base_wait * (2 ** attempt)
            print(f"⏳ Esperando {wait:.1f}s antes de reintentar...")
            await asyncio.sleep(wait)

    raise last_exc
