"""
ai.py — the ONE gateway between retailAPOLLO and the firm's LLM.
================================================================

Every AI feature in this project (the AI Pulse page, the agentic-watch
digest, the keyword-map auditor, notebook 10's sentiment test) talks to
the model through this module and nothing else, so the connection details
live in exactly one place.

Connection: the firm gateway is reached through `dimsum_lite`'s
Apollo-authenticated OpenAI factory —

    from dimsum_lite.clients.openai import ApolloOpenAI
    apollo = ApolloOpenAI(env=ENVIRONMENT)      # auth from env vars
    client = apollo.client()                    # a standard OpenAI client
    client.chat.completions.create(model="gpt-4o", messages=[...])

Auth is handled by dimsum_lite from ENVIRONMENT / APOLLO_AUTH_USERNAME /
APOLLO_AUTH_PASSWORD (the project's .env is loaded first, so the same
file that holds the FetchLayer key holds these).  The gateway needs the
VPN and the JFrog-installed package, so IT ONLY WORKS ON THE DESK'S OWN
MACHINE — everywhere else `available()` is False and every caller is
expected to degrade politely (samples, PENDING banners) instead of
crashing.  A round-trip measured ~4.6s, so callers batch: few calls,
big payloads.

CONFIG (all optional, all read from .env / the environment):
    ENVIRONMENT              Apollo environment ('DEV', 'UAT', ...)
    APOLLO_AUTH_USERNAME     defaults to the OS user
    APOLLO_AUTH_PASSWORD     prompted by dimsum_lite if absent
    AI_MODEL                 deployment name, default 'gpt-4o' — swap for
                             one your env exposes (dimsum_lite.constants
                             lists them; 'model-not-found' means this)
    AI_DATA_CLASSIFICATION   'PUBLIC'|'RESTRICTED'|'CONFIDENTIAL'|'MNPI',
                             default 'RESTRICTED' (posts are public text;
                             RESTRICTED is the conservative default)
    AI_USER_ID               passed to apollo.client() if set
    AI_MAX_CALLS             hard per-process budget, default 80 — a
                             runaway loop hits this, never the gateway.
                             One full update spends about 36: the poll's
                             30 prompts, the pulse's 5 calls and the
                             weekly keyword audit, with headroom so a
                             couple of retries cannot exhaust it
    AI_MOCK                  '1' = return deterministic canned output
                             without any network (tests, cloud dev)

Self-test:
    python -m src.ai --selftest
"""

from __future__ import annotations

import json
import os
import re
import time

# the project .env, same loading convention as the fetchers
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_env() -> None:
    path = os.path.join(_ROOT, ".env")
    if not os.path.exists(path):
        return
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip()
        if k and v and k not in os.environ:
            os.environ[k] = v


_load_env()

MODEL = os.environ.get("AI_MODEL", "gpt-4o")
MAX_CALLS = int(os.environ.get("AI_MAX_CALLS", "80"))
MOCK = os.environ.get("AI_MOCK", "") == "1"

_client = None
_unavailable_reason: str | None = None
_calls_made = 0


def _connect():
    """Build the Apollo OpenAI client once. Never raises — records why
    the gateway is unreachable instead, so a dashboard render or a
    notebook run far from the VPN stays alive."""
    global _client, _unavailable_reason
    if _client is not None or _unavailable_reason is not None:
        return _client
    if MOCK:
        _unavailable_reason = None
        return None                       # mock path answers without one
    try:
        from dimsum_lite.clients.openai import ApolloOpenAI  # noqa: PLC0415
    except Exception as e:                                   # noqa: BLE001
        _unavailable_reason = (f"dimsum_lite not importable ({e}) - the "
                               "Apollo gateway only exists on the desk "
                               "machine (JFrog/LEVA install)")
        return None
    try:
        apollo = ApolloOpenAI(env=os.environ.get("ENVIRONMENT", "DEV"))
        kwargs = {}
        if os.environ.get("AI_USER_ID"):
            kwargs["user_id"] = os.environ["AI_USER_ID"]
        kwargs["data_classification"] = os.environ.get(
            "AI_DATA_CLASSIFICATION", "RESTRICTED")
        try:
            _client = apollo.client(**kwargs)
        except TypeError:
            # older dimsum_lite: client() takes no kwargs
            _client = apollo.client()
    except Exception as e:                                   # noqa: BLE001
        _unavailable_reason = (f"Apollo auth/connection failed "
                               f"({type(e).__name__}: {e}) - check "
                               "ENVIRONMENT, APOLLO_AUTH_USERNAME/"
                               "PASSWORD and the VPN")
        return None
    return _client


def available() -> bool:
    """True when a call to chat() can be expected to work."""
    if MOCK:
        return True
    return _connect() is not None


def explain_unavailable() -> str:
    """One sentence for a PENDING banner."""
    _connect()
    return _unavailable_reason or "gateway reachable"


def chat(prompt: str, system: str | None = None, *,
         want_json: bool = False, max_tokens: int = 1800,
         temperature: float = 0.0, retries: int = 2):
    """One completion. Returns str (or parsed object when want_json).
    Raises RuntimeError when the gateway is unreachable or the per-run
    call budget (AI_MAX_CALLS) is spent — callers that must not crash
    check available() first."""
    global _calls_made
    if _calls_made >= MAX_CALLS:
        raise RuntimeError(f"AI_MAX_CALLS budget ({MAX_CALLS}) spent - "
                           "raise it in .env if this run is legitimate")
    if MOCK:
        _calls_made += 1
        return _mock_answer((system or "") + "\n" + prompt, want_json)
    client = _connect()
    if client is None:
        raise RuntimeError(f"LLM unavailable: {_unavailable_reason}")
    messages = ([{"role": "system", "content": system}] if system else [])
    messages.append({"role": "user", "content": prompt})
    last = None
    for attempt in range(retries + 1):
        try:
            _calls_made += 1
            resp = client.chat.completions.create(
                model=MODEL, messages=messages,
                max_tokens=max_tokens, temperature=temperature)
            text = resp.choices[0].message.content or ""
            return _parse_json(text) if want_json else text
        except Exception as e:                               # noqa: BLE001
            last = e
            if attempt < retries:
                time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"LLM call failed after {retries + 1} tries: "
                       f"{type(last).__name__}: {last}")


def _parse_json(text: str):
    """The gateway's deployments do not all honour response_format, so
    JSON is asked for in the prompt and extracted defensively here."""
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if m:
        text = m.group(1).strip()
    start = min((i for i in (text.find("{"), text.find("["))
                 if i >= 0), default=0)
    return json.loads(text[start:])


# ---------------------------------------------------------------------------
# MOCK — deterministic offline answers so the whole plumbing (pulse
# generation, dashboard rendering, notebooks, unit tests) runs end to end
# with no gateway.  Clearly labelled in every output.
# ---------------------------------------------------------------------------
def _mock_answer(prompt: str, want_json: bool):
    if not want_json:
        if "JSON:" in prompt or "retail investor" in prompt.lower():
            return _mock_poll_answer()
        return ("[MOCK - no gateway] A deterministic placeholder "
                "answer for offline testing.")
    # Shape-matching mocks for the known JSON consumers. ORDER MATTERS:
    # the pulse makes four separate JSON calls and each is recognised by
    # a key that only IT asks for, most specific first.
    if "market_vibe" in prompt:
        return {
            "market_vibe": {
                "bullets": ["[MOCK] Offline placeholder vibe line."],
                "one_liner": "[MOCK] placeholder line - no gateway.",
                "one_liner_why": "[MOCK] Placeholder.",
            },
            "mood_gauge": {"score": 50, "why": "[MOCK] Placeholder."},
            "market_pulse": "[MOCK] Offline placeholder pulse - run on "
                            "the desk machine (VPN + dimsum_lite) for "
                            "the real one.",
            "talk_of_the_town": "[MOCK] Placeholder.",
        }
    if "theme_briefs" in prompt:
        return {"theme_briefs": []}
    if "catalyst_watch" in prompt or "divergences" in prompt:
        return {"catalyst_watch": [], "divergences": []}
    if "market_pulse" in prompt:               # any other caller
        return {
            "market_pulse": "[MOCK] Offline placeholder pulse.",
            "talk_of_the_town": "[MOCK] Placeholder.",
            "mood_gauge": {"score": 50, "why": "[MOCK] Placeholder."},
            "theme_briefs": [],
            "catalyst_watch": [], "divergences": [],
        }
    if "agentic" in prompt.lower():
        return {"digest": "[MOCK] Placeholder agentic digest.",
                "asks": [], "actions": [], "risk_note": "[MOCK]"}
    if "keyword" in prompt.lower():
        return {"additions": [], "moves": [], "removals": [],
                "notes": "[MOCK] no suggestions offline"}
    if "sentiment" in prompt.lower():
        return [{"i": 0, "label": "neutral", "score": 0.0}]
    return {"mock": True}


def _mock_poll_answer() -> str:
    return ('[MOCK - no gateway] I cannot give real recommendations '
            'offline.\nJSON: {"tickers": [{"symbol": "MOCK", '
            '"direction": "buy", "conviction": "low"}], '
            '"themes": ["mock"], "summary": "[MOCK] offline placeholder"}')


def calls_made() -> int:
    return _calls_made


# ---------------------------------------------------------------------------
def _selftest() -> int:
    """Verifies gateway connectivity and prints the effective config."""
    print(f"model={MODEL}  mock={MOCK}  "
          f"env={os.environ.get('ENVIRONMENT', '(unset)')}")
    if not available():
        print(f"[FAIL] {explain_unavailable()}")
        return 1
    t0 = time.time()
    answer = chat("What is the capital of France?")
    dt = time.time() - t0
    print(f"Q: What is the capital of France?\nA: {answer}")
    ok = "paris" in str(answer).lower() or MOCK
    print(f"[{'OK' if ok else 'FAIL'}] round-trip {dt:.2f}s")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        raise SystemExit(_selftest())
    print(__doc__)
