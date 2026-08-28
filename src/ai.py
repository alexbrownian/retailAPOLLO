"""
ai.py — the ONE gateway between retailAPOLLO and the firm's LLM.
================================================================

Every AI feature in this project (the AI Pulse page, the agentic-watch
digest, the keyword-map auditor, notebook 10's sentiment test) talks to
the model through this module and nothing else, so the connection details
live in exactly one place.

TWO PROVIDERS, ONE INTERFACE
    apollo     the firm gateway, reached through `dimsum_lite`'s
               Apollo-authenticated OpenAI factory —

                   from dimsum_lite.clients.openai import ApolloOpenAI
                   apollo = ApolloOpenAI(env=ENVIRONMENT)
                   client = apollo.client()     # a standard OpenAI client
                   client.chat.completions.create(model=..., messages=[...])

               Auth is handled by dimsum_lite from ENVIRONMENT /
               APOLLO_AUTH_USERNAME / APOLLO_AUTH_PASSWORD.  It needs the
               VPN and the JFrog-installed package, so it only resolves on
               the desk machine.

    anthropic  the Claude API direct, on an ANTHROPIC_API_KEY.  Needs
               nothing but the key and a network route, which is what
               makes a personal machine a complete environment: the AI
               Pulse and the agentic digest regenerate off the VPN
               instead of degrading to PENDING banners.

`AI_PROVIDER` picks between them and defaults to 'auto': try Apollo,
fall back to Anthropic, and report both reasons if neither resolves.
Selection happens once per process and is visible via `provider()`, so a
banner can say which model actually answered.  When neither resolves
`available()` is False and every caller is expected to degrade politely
(samples, PENDING banners) instead of crashing.  An Apollo round-trip
measured ~4.6s, so callers batch: few calls, big payloads.

CONFIG (all optional, all read from .env / the environment):
    AI_PROVIDER              'auto' (default) | 'apollo' | 'anthropic'.
                             Naming one skips the other entirely, which
                             is how a machine that could reach both is
                             pinned to the cheaper or the approved one
    ENVIRONMENT              Apollo environment ('DEV', 'UAT', ...)
    APOLLO_AUTH_USERNAME     defaults to the OS user
    APOLLO_AUTH_PASSWORD     prompted by dimsum_lite if absent
    ANTHROPIC_API_KEY        enables the Anthropic provider; absent means
                             that provider simply never resolves
    ANTHROPIC_MODEL          model id, default 'claude-sonnet-5'
    AI_MODEL                 Apollo deployment name, default 'gpt-4o' —
                             swap for one your env exposes
                             (dimsum_lite.constants lists them;
                             'model-not-found' means this)
    AI_DATA_CLASSIFICATION   'PUBLIC'|'RESTRICTED'|'CONFIDENTIAL'|'MNPI',
                             default 'RESTRICTED' (posts are public text;
                             RESTRICTED is the conservative default)
    AI_USER_ID               passed to apollo.client() if set
    AI_MAX_CALLS             hard per-process budget, default 80 — a
                             runaway loop hits this, never a provider.
                             A full update now spends about 40: the
                             poll's 30 prompts, the pulse's 9 (one
                             whole-market read, SIX theme-brief batches,
                             catalysts, agentic) and the weekly keyword
                             audit. The pulse's batch count is not fixed
                             — it is ceil(themes / THEMES_PER_CALL), so
                             halving that constant doubles those calls.
                             It was halved to 6 when a more verbose model
                             began truncating 12-theme batches, which
                             took a run from ~37 calls to exactly 40 and
                             silently exhausted a 40-call budget on the
                             LAST call of the pulse. Leave real headroom:
                             a budget sized to the expected cost fails
                             the moment one call retries
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
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")
PROVIDER_PREF = os.environ.get("AI_PROVIDER", "auto").strip().lower()
MAX_CALLS = int(os.environ.get("AI_MAX_CALLS", "80"))
MOCK = os.environ.get("AI_MOCK", "") == "1"

_client = None
_provider: str | None = None
_unavailable_reason: str | None = None
_calls_made = 0


def _connect_apollo():
    """Builds the Apollo OpenAI client. Returns (client, reason)."""
    try:
        from dimsum_lite.clients.openai import ApolloOpenAI  # noqa: PLC0415
    except Exception as e:                                   # noqa: BLE001
        return None, (f"dimsum_lite not importable ({e}) - the Apollo "
                      "gateway only exists on the desk machine "
                      "(JFrog/LEVA install)")
    try:
        apollo = ApolloOpenAI(env=os.environ.get("ENVIRONMENT", "DEV"))
        kwargs = {}
        if os.environ.get("AI_USER_ID"):
            kwargs["user_id"] = os.environ["AI_USER_ID"]
        kwargs["data_classification"] = os.environ.get(
            "AI_DATA_CLASSIFICATION", "RESTRICTED")
        try:
            client = apollo.client(**kwargs)
        except TypeError:
            # older dimsum_lite: client() takes no kwargs
            client = apollo.client()
    except Exception as e:                                   # noqa: BLE001
        return None, (f"Apollo auth/connection failed "
                      f"({type(e).__name__}: {e}) - check ENVIRONMENT, "
                      "APOLLO_AUTH_USERNAME/PASSWORD and the VPN")
    return client, None


def _connect_anthropic():
    """Builds the Anthropic client. Returns (client, reason).

    Constructing the client does not call the network, so a wrong key
    surfaces at the first chat() rather than here.
    """
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key:
        return None, "no ANTHROPIC_API_KEY in .env or the environment"
    try:
        import anthropic                                     # noqa: PLC0415
    except Exception as e:                                   # noqa: BLE001
        return None, (f"anthropic package not installed ({e}) - "
                      "pip install anthropic")
    try:
        # HARD TIMEOUT (defect report: a network that silently drops
        # traffic to the endpoint accepts the connection and then never
        # answers - the SDK's default 10-minute timeout made every
        # attempt look like a hang. 60s is generous for a real answer
        # and turns a black-hole network into a clear error in a
        # minute, not half an hour of retries. max_retries=0: chat()
        # already does its own retrying, the SDK doubling it quadrupled
        # the wait.
        return anthropic.Anthropic(api_key=key, timeout=60.0,
                                   max_retries=0), None
    except Exception as e:                                   # noqa: BLE001
        return None, f"Anthropic client init failed ({type(e).__name__}: {e})"


_BUILDERS = {"apollo": _connect_apollo, "anthropic": _connect_anthropic}


def _resolution_order() -> list:
    """Providers to try, in order, for the configured preference."""
    if PROVIDER_PREF in _BUILDERS:
        return [PROVIDER_PREF]
    return ["apollo", "anthropic"]        # 'auto': firm gateway first


def _connect():
    """Resolves a provider once. Never raises — records why every
    candidate failed instead, so a dashboard render or a notebook run
    far from the VPN stays alive."""
    global _client, _provider, _unavailable_reason
    if _client is not None or _unavailable_reason is not None:
        return _client
    if MOCK:
        _unavailable_reason = None
        return None                       # mock path answers without one
    reasons = []
    for kind in _resolution_order():
        client, reason = _BUILDERS[kind]()
        if client is not None:
            _client, _provider = client, kind
            return _client
        reasons.append(f"{kind}: {reason}")
    _unavailable_reason = "; ".join(reasons)
    return None


def provider() -> str | None:
    """Which provider answered, once one has resolved ('mock' under
    AI_MOCK). None when nothing is reachable — for banners that name the
    model behind a generated block."""
    if MOCK:
        return "mock"
    _connect()
    return _provider


def active_model() -> str | None:
    """The model id the resolved provider will be called with."""
    kind = provider()
    if kind == "anthropic":
        return ANTHROPIC_MODEL
    if kind == "apollo":
        return MODEL
    return None


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
         want_json: bool = False, max_tokens: int = 4000,
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
    last = None
    for attempt in range(retries + 1):
        try:
            _calls_made += 1
            text = (_call_anthropic(client, prompt, system, max_tokens,
                                    temperature)
                    if _provider == "anthropic"
                    else _call_openai(client, prompt, system, max_tokens,
                                      temperature))
            return _parse_json(text) if want_json else text
        except Exception as e:                               # noqa: BLE001
            last = e
            if attempt < retries:
                time.sleep(2 * (attempt + 1))
    _hint = ""
    if type(last).__name__ == "APIConnectionError":
        # the SDK's str() is often empty here - the real reason (DNS,
        # proxy, TLS interception) lives in the cause chain
        _cause = getattr(last, "__cause__", None)
        if _cause is not None:
            _hint += f" | cause: {type(_cause).__name__}: {_cause}"
        _hint += (" | the machine could not reach the endpoint at all: "
                  "check the network/VPN (a corporate proxy that "
                  "intercepts TLS needs `pip install pip-system-certs` "
                  "in THIS venv), and that the provider host is "
                  "reachable from this machine")
    raise RuntimeError(f"LLM call failed after {retries + 1} tries: "
                       f"{type(last).__name__}: {last}{_hint}")


def _call_openai(client, prompt, system, max_tokens, temperature) -> str:
    """One Apollo/OpenAI chat completion, returned as text."""
    messages = ([{"role": "system", "content": system}] if system else [])
    messages.append({"role": "user", "content": prompt})
    resp = client.chat.completions.create(
        model=MODEL, messages=messages,
        max_tokens=max_tokens, temperature=temperature)
    return resp.choices[0].message.content or ""


_accepts_temperature: bool | None = None


def _anthropic_accepts_temperature(client) -> bool:
    """Whether the installed SDK's messages.create() takes `temperature`.

    The v1 SDK removed the parameter; v0.x accepts it. Passing it to v1
    raises TypeError before any request is made, so the capability is
    probed once from the signature rather than discovered per call.
    """
    global _accepts_temperature
    if _accepts_temperature is None:
        import inspect                                       # noqa: PLC0415
        try:
            _accepts_temperature = "temperature" in inspect.signature(
                client.messages.create).parameters
        except (TypeError, ValueError):
            _accepts_temperature = False
    return _accepts_temperature


def _call_anthropic(client, prompt, system, max_tokens, temperature) -> str:
    """One Anthropic message, returned as text.

    The Messages API takes the system prompt as a top-level argument
    rather than a leading message, and answers with a list of content
    blocks; the text blocks are concatenated so a response split across
    several arrives whole.

    `temperature` is forwarded only where the SDK still accepts it. On v1
    it is dropped: callers that raise it for variety (the poll's 0.8) get
    the model's default sampling instead, so answers to one prompt vary
    less than on the gateway. Variety across the poll's 30 prompts comes
    from the prompts themselves and is unaffected.
    """
    kwargs = {"model": ANTHROPIC_MODEL, "max_tokens": max_tokens,
              "messages": [{"role": "user", "content": prompt}]}
    if system:
        kwargs["system"] = system
    if _anthropic_accepts_temperature(client):
        kwargs["temperature"] = temperature
    resp = client.messages.create(**kwargs)
    text = "".join(block.text for block in resp.content
                   if getattr(block, "type", None) == "text")
    if not text.strip():
        # An empty answer used to be returned as "", which _parse_json
        # then reported as `Expecting value: line 1 column 1 (char 0)` -
        # a message that names the symptom and hides every cause. Say
        # what the API actually reported instead: stop_reason
        # distinguishes a refusal from a truncation from an empty turn,
        # and the block types show whether the text simply arrived in a
        # shape this join does not read.
        raise RuntimeError(
            "the model returned no text "
            f"(stop_reason={getattr(resp, 'stop_reason', '?')!r}, "
            f"blocks={[getattr(b, 'type', '?') for b in resp.content]}, "
            f"max_tokens={max_tokens})")
    if getattr(resp, "stop_reason", None) == "max_tokens":
        # Truncated JSON parses as a syntax error somewhere in the
        # middle, which reads like a model fault rather than a budget
        # one. Name it at the point it happens.
        raise RuntimeError(
            f"the answer hit the {max_tokens}-token ceiling and was cut "
            "off mid-sentence; raise max_tokens for this call or ask "
            "for fewer items at once")
    return text


def _parse_json(text: str):
    """The gateway's deployments do not all honour response_format, so
    JSON is asked for in the prompt and extracted defensively here.

    strict=False is deliberate. A model writing a paragraph into a JSON
    string value puts REAL newlines and tabs inside the quotes rather
    than the \\n escapes the spec demands, and strict parsing rejects
    the whole document for it:

        JSONDecodeError: Invalid control character at: line 21 column 1151

    That is a formatting nicety, not a corrupt answer - the text either
    side of it is exactly what was asked for - so the control characters
    are accepted rather than the response thrown away. Everything else
    about the parse stays strict: a genuinely malformed object still
    raises, and the caller still retries.
    """
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if m:
        text = m.group(1).strip()
    start = min((i for i in (text.find("{"), text.find("["))
                 if i >= 0), default=0)
    body = text[start:]
    try:
        return json.loads(body, strict=False)
    except json.JSONDecodeError:
        # LAST RESORT: a trailing comma or an unterminated tail from a
        # long answer. Walk back to the last balanced close and try that
        # prefix, so one ragged ending does not discard a good object.
        depth, last_ok = 0, None
        for i, ch in enumerate(body):
            if ch in "{[":
                depth += 1
            elif ch in "}]":
                depth -= 1
                if depth == 0:
                    last_ok = i + 1
        if last_ok:
            return json.loads(body[:last_ok], strict=False)
        raise


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
    """Verifies provider connectivity and prints the effective config."""
    print(f"preference={PROVIDER_PREF}  mock={MOCK}  "
          f"env={os.environ.get('ENVIRONMENT', '(unset)')}")
    print(f"apollo model={MODEL}  anthropic model={ANTHROPIC_MODEL}")
    if not available():
        print(f"[FAIL] {explain_unavailable()}")
        return 1
    print(f"resolved provider={provider()}  model={active_model()}")
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
