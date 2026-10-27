"""One ask_llm() interface over several OpenAI-compatible providers.

Primary: Google Gemini (OpenAI-compatible endpoint). Fallbacks: Groq, then a local Ollama model.
On 429 / network errors / timeouts: one short backoff retry, then the next provider.
Every call is counted (self.calls) so tests can assert "zero LLM calls".
"""
from __future__ import annotations

import base64
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, Iterable, List, Optional, Tuple

log = logging.getLogger(__name__)

try:  # the app must still run (no-LLM mode) if openai is missing
    import openai
    from openai import OpenAI
except Exception:  # pragma: no cover
    openai = None
    OpenAI = None

GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/openai/"
GROQ_BASE = "https://api.groq.com/openai/v1"
OLLAMA_BASE = "http://localhost:11434/v1"
OPENROUTER_BASE = "https://openrouter.ai/api/v1"

# gemini-flash-latest is Google's auto-updating alias for the current Flash model, so it keeps
# working when a numbered model is retired (gemini-2.5-flash was shut down in October 2026).
DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
DEFAULT_GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
DEFAULT_OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openrouter/free")


@dataclass
class Provider:
    name: str
    base_url: str
    api_key: str
    model: str
    vision: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return f"{self.name} · {self.model}"


class LLMError(RuntimeError):
    pass


def build_providers(gemini_key: str = "", gemini_model: str = "", groq_key: str = "", groq_model: str = "",
                    use_ollama: bool = False, ollama_model: str = "", ollama_url: str = "",
                    openrouter_key: str = "", openrouter_model: str = "") -> List[Provider]:
    providers: List[Provider] = []
    if gemini_key:
        providers.append(Provider("Gemini", GEMINI_BASE, gemini_key.strip(), (gemini_model or DEFAULT_GEMINI_MODEL).strip(),
                                  vision=True, extra={"reasoning_effort": os.getenv("GEMINI_REASONING_EFFORT", "low")}))
    if groq_key:
        providers.append(Provider("Groq", GROQ_BASE, groq_key.strip(), (groq_model or DEFAULT_GROQ_MODEL).strip()))
    if openrouter_key:
        providers.append(Provider("OpenRouter", OPENROUTER_BASE, openrouter_key.strip(),
                                  (openrouter_model or DEFAULT_OPENROUTER_MODEL).strip()))
    if use_ollama:
        providers.append(Provider("Ollama", (ollama_url or OLLAMA_BASE).strip(), "ollama",
                                  (ollama_model or DEFAULT_OLLAMA_MODEL).strip()))
    return providers


def providers_from_env() -> List[Provider]:
    return build_providers(
        gemini_key=os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", ""),
        gemini_model=os.getenv("GEMINI_MODEL", ""),
        groq_key=os.getenv("GROQ_API_KEY", ""),
        groq_model=os.getenv("GROQ_MODEL", ""),
        use_ollama=os.getenv("USE_OLLAMA", "").lower() in ("1", "true", "yes"),
        ollama_model=os.getenv("OLLAMA_MODEL", ""),
        ollama_url=os.getenv("OLLAMA_BASE_URL", ""),
        openrouter_key=os.getenv("OPENROUTER_API_KEY", ""),
        openrouter_model=os.getenv("OPENROUTER_MODEL", ""),
    )


def _is_retryable(exc: Exception) -> bool:
    if openai is None:
        return False
    retry_types = tuple(t for t in (getattr(openai, "RateLimitError", None), getattr(openai, "APIConnectionError", None),
                                    getattr(openai, "APITimeoutError", None), getattr(openai, "InternalServerError", None))
                        if t is not None)
    return isinstance(exc, retry_types)


def _is_bad_request(exc: Exception) -> bool:
    return openai is not None and isinstance(exc, getattr(openai, "BadRequestError", ()))


def _short_error(exc: Exception) -> str:
    msg = str(exc)
    msg = re.sub(r"\s+", " ", msg)
    status = getattr(exc, "status_code", None)
    return f"{type(exc).__name__}{f' {status}' if status else ''}: {msg[:220]}"


class LLMClient:
    """Thin wrapper with provider fallback, retry/backoff, call counting and a call log."""

    def __init__(self, providers: List[Provider], timeout: float = 30.0, max_tokens: int = 450,
                 backoff: float = 1.5):
        self.providers = [p for p in providers if p.api_key]
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.backoff = backoff
        self.calls = 0
        self.log: List[Dict[str, Any]] = []
        self.last_provider: Optional[str] = None
        self._clients: Dict[str, Any] = {}

    # ------------------------------------------------------------------ helpers
    @property
    def available(self) -> bool:
        return bool(self.providers) and OpenAI is not None

    def _client(self, p: Provider):
        key = f"{p.name}|{p.base_url}|{p.api_key[-6:]}"
        if key not in self._clients:
            self._clients[key] = OpenAI(api_key=p.api_key, base_url=p.base_url, timeout=self.timeout, max_retries=0)
        return self._clients[key]

    def _record(self, p: Provider, ok: bool, started: float, note: str = "") -> None:
        self.log.append({"provider": p.name, "model": p.model, "ok": ok,
                         "seconds": round(time.time() - started, 2), "note": note, "at": time.strftime("%H:%M:%S")})

    def _kwargs(self, p: Provider, messages: List[Dict[str, Any]], max_tokens: int, json_mode: bool,
                stream: bool) -> Dict[str, Any]:
        kw: Dict[str, Any] = {"model": p.model, "messages": messages, "temperature": 0, "max_tokens": max_tokens}
        if stream:
            kw["stream"] = True
        if json_mode:
            kw["response_format"] = {"type": "json_object"}
        if p.extra.get("reasoning_effort"):
            kw["reasoning_effort"] = p.extra["reasoning_effort"]
        return kw

    # ------------------------------------------------------------------ non-streaming
    def chat(self, messages: List[Dict[str, Any]], max_tokens: Optional[int] = None, json_mode: bool = False,
             vision: bool = False) -> Tuple[str, str]:
        """Return (text, provider_label). Raises LLMError if every provider fails."""
        if not self.available:
            raise LLMError("no LLM provider configured")
        errors: List[str] = []
        providers = [p for p in self.providers if p.vision] if vision else self.providers
        if not providers:
            raise LLMError("no vision-capable provider configured (Gemini key needed)")
        for p in providers:
            kw = self._kwargs(p, messages, max_tokens or self.max_tokens, json_mode, stream=False)
            attempt = 0
            while attempt < 4:
                attempt += 1
                started = time.time()
                try:
                    self.calls += 1
                    resp = self._client(p).chat.completions.create(**kw)
                    choice = resp.choices[0] if resp.choices else None
                    text = (choice.message.content or "") if choice and choice.message else ""
                    if not text.strip():
                        # Thinking models can spend the whole token budget on reasoning -> retry with room
                        self._record(p, False, started, "empty reply")
                        if kw["max_tokens"] < 2000:
                            kw["max_tokens"] = 2000
                            kw.pop("reasoning_effort", None)
                            continue
                        errors.append(f"{p.name}: empty reply")
                        break
                    self._record(p, True, started)
                    self.last_provider = p.label
                    return text.strip(), p.label
                except Exception as exc:  # noqa: BLE001 - we classify below
                    self._record(p, False, started, _short_error(exc))
                    if _is_bad_request(exc) and ("reasoning_effort" in kw or "response_format" in kw):
                        # provider/model does not support an optional parameter: drop it and retry
                        kw.pop("reasoning_effort", None) if "reasoning_effort" in kw else kw.pop("response_format", None)
                        continue
                    if _is_retryable(exc) and attempt == 1:
                        time.sleep(self.backoff)
                        continue
                    errors.append(f"{p.name}: {_short_error(exc)}")
                    break
        raise LLMError("All LLM providers failed — " + " | ".join(errors))

    # ------------------------------------------------------------------ streaming
    def stream(self, messages: List[Dict[str, Any]], max_tokens: Optional[int] = None) -> Generator[str, None, None]:
        """Yield text chunks; falls back across providers if a stream cannot start.

        After the generator finishes, self.last_provider names who answered.
        """
        if not self.available:
            raise LLMError("no LLM provider configured")
        errors: List[str] = []
        for p in self.providers:
            kw = self._kwargs(p, messages, max_tokens or self.max_tokens, False, stream=True)
            attempt = 0
            while attempt < 4:
                attempt += 1
                started = time.time()
                got_any = False
                try:
                    self.calls += 1
                    stream = self._client(p).chat.completions.create(**kw)
                    for chunk in stream:
                        if not getattr(chunk, "choices", None):
                            continue
                        delta = chunk.choices[0].delta
                        piece = getattr(delta, "content", None) if delta is not None else None
                        if piece:
                            got_any = True
                            yield piece
                    if not got_any:
                        self._record(p, False, started, "empty stream")
                        if kw["max_tokens"] < 2000:
                            kw["max_tokens"] = 2000
                            kw.pop("reasoning_effort", None)
                            continue
                        errors.append(f"{p.name}: empty reply")
                        break
                    self._record(p, True, started)
                    self.last_provider = p.label
                    return
                except Exception as exc:  # noqa: BLE001
                    self._record(p, False, started, _short_error(exc))
                    if got_any:
                        self.last_provider = p.label + " (stream interrupted)"
                        yield "\n\n_(answer interrupted by a network error)_"
                        return
                    if _is_bad_request(exc) and "reasoning_effort" in kw:
                        kw.pop("reasoning_effort", None)
                        continue
                    if _is_retryable(exc) and attempt == 1:
                        time.sleep(self.backoff)
                        continue
                    errors.append(f"{p.name}: {_short_error(exc)}")
                    break
        raise LLMError("All LLM providers failed — " + " | ".join(errors))

    # ------------------------------------------------------------------ convenience
    def chat_json(self, system: str, user: str, max_tokens: int = 900) -> Tuple[Dict[str, Any], str]:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        text, label = self.chat(messages, max_tokens=max_tokens, json_mode=True)
        return parse_json_loose(text), label

    def vision_json(self, system: str, prompt: str, png_bytes: bytes, max_tokens: int = 1200) -> Tuple[Dict[str, Any], str]:
        b64 = base64.b64encode(png_bytes).decode("ascii")
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ]},
        ]
        text, label = self.chat(messages, max_tokens=max_tokens, json_mode=True, vision=True)
        return parse_json_loose(text), label

    def test_connection(self) -> List[Dict[str, str]]:
        """Ping each provider separately with a tiny prompt (used by the sidebar 'Test LLM' button)."""
        results = []
        for p in self.providers:
            single = LLMClient([p], timeout=min(self.timeout, 20), max_tokens=20, backoff=0.5)
            try:
                text, _ = single.chat([{"role": "user", "content": "Reply with the single word OK."}], max_tokens=20)
                results.append({"provider": p.label, "status": "ok", "detail": text[:40]})
            except Exception as exc:  # noqa: BLE001
                results.append({"provider": p.label, "status": "failed", "detail": str(exc)[:300]})
            self.calls += single.calls
        return results


def parse_json_loose(text: str) -> Dict[str, Any]:
    """Parse a JSON object from a model reply that may be wrapped in ```json fences or prose."""
    if not text:
        return {}
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.IGNORECASE | re.MULTILINE).strip()
    try:
        obj = json.loads(t)
        return obj if isinstance(obj, dict) else {"items": obj}
    except Exception:
        pass
    m = re.search(r"\{.*\}", t, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}
    return {}
