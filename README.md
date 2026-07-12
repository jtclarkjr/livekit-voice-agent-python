# LiveKit Voice Agent (Python)

Deployment-ready async Python worker for ChatGPT-style voice sessions. It is an explicitly dispatched LiveKit Agent named
`realtime-chat-voice`. LiveKit Inference provides speech-to-text,
language-model, and text-to-speech processing; the existing Elixir AI service is
not used.

## Runtime behavior

- Loads up to ten existing personal-chat messages from `api` before
  joining the room.
- Before each model call, builds an inference-only window containing the ten
  most recent user/assistant messages, the current turn's sanitized tool
  observations, and the current UTC timestamp. Older tool results are removed
  from the prompt while canonical session history remains available for final
  transcript reconciliation.
- Waits in listening mode and never generates an automatic greeting.
- Uses `deepgram/nova-3` with multilingual recognition,
  `google/gemma-4-31b-it`, and `cartesia/sonic-3.5` with full-band 48 kHz mono
  synthesis. The backend selects one allowlisted voice key: `jacqueline`
  (`9626c31c-bec5-4cca-baa8-f8ba9e84c8bc`), `blake`
  (`a167e0f3-df7e-4d52-a9c3-f949145efdab`), or `robyn`
  (`f31cc6a7-c1e8-4764-980c-60a361443dd1`).
- Uses LiveKit's audio turn detector, preemptive generation, and Quail Voice
  Focus noise cancellation. Parallel model tool calls are disabled.
- Persists committed user and assistant items to `api` in order through
  one job-scoped `httpx.AsyncClient` and a single-consumer `asyncio.Queue`.
  Stable LiveKit item IDs and session-local sequence numbers make retries
  idempotent.
- Reconciles the complete final LiveKit chat history on shutdown, including
  interrupted assistant text. A finalization failure falls back to the failure
  endpoint with the best available partial transcript, and settlement occurs at
  most once.
- Suppresses duplicate jobs for the same voice session and keeps a linked
  browser participant alive through a ten-second refresh/rejoin grace window.
  Explicit backend end requests still stop the worker immediately.
- Starts `AgentSession` with `record=False`, disables automatic closure on
  participant disconnect, and opts out of LiveKit Agent Insights uploads for
  audio, transcripts, traces, and logs.

Startup ordering is deliberate: fetch and validate bridge context, start the
LiveKit session, exclude cloned history item IDs, register live persistence,
connect the worker, and only then mark the backend session started.

Typed in-call messages use LiveKit's session transport and enter the same
committed history as speech. This repository never records or stores audio and
never sends audio to `api`.

## Web-search policy and privacy

The deterministic `searchWeb` policy gives an explicit opt-out priority over
every other signal. A confirmed request to browse, verify, or answer a question
whose information may have changed requires exactly one search. Short
follow-ups may inherit time-sensitive intent for a bounded window. Duplicate
tool calls are rejected, and interruption cancels the outbound request.
Otherwise the model can search only when it is genuinely uncertain.

Search crosses the LiveKit zero-retention boundary: the minimum necessary query
is sent to Tavily. Provider metadata, URLs, and unsafe fields are stripped; only
sanitized snippets may reach inference memory. Queries, snippets, tool calls,
tool outputs, source names, and URLs are never persisted to `api` and are
discarded with the in-memory session. Only final user and assistant turns enter
the durable transcript. Responses do not expose source names, citations, or
URLs.

`LOG_LEVEL` controls LiveKit operational logs separately. It defaults to
`fatal`, which Python maps to `critical`, because lower log levels may contain
valid or malformed tool arguments. Lowering it is an explicit operational
choice that can expose search queries to log retention. Never place private
conversation details, secrets, or identifiers in a search query.

Preemptive generation can begin speculative inference from partial speech
before end-of-turn confirmation. If the final transcript changes, that
inference is cancelled or discarded even though model tokens may already have
been consumed. Tool execution waits for turn confirmation, so Tavily is not
called from provisional speech.

## Prompt resources

All model-facing instructions, runtime-policy fragments, query guidance, and
tool descriptions live as Markdown files in `src/agent/prompts/`. Python code
selects and renders those resources through `agent.prompt_loader`; prompt text
does not live in constants or decorator string literals. Template variables use
the explicit `{{variable_name}}` form and must match exactly at render time.

## Configuration

Copy `.env.example` to `.env.local` and provide the required server-side values:

| Variable | Purpose |
| --- | --- |
| `LIVEKIT_URL` | LiveKit Cloud WebSocket URL. |
| `LIVEKIT_API_KEY` / `LIVEKIT_API_SECRET` | Server-side LiveKit credentials and Inference authorization. |
| `LIVEKIT_AGENT_NAME` | Must remain `realtime-chat-voice` to match explicit dispatch. |
| `API_URL` | Root backend API URL, for example `http://localhost:4000`. |
| `VOICE_AGENT_BRIDGE_TOKEN` | Shared Bearer secret for internal voice endpoints. |
| `TAVILY_API_KEY` | Server-side Tavily credential for transient web search. |
| `VOICE_WEB_SEARCH_MAX_RESULTS` | Optional result limit; defaults to `5`. |
| `VOICE_WEB_SEARCH_TIMEOUT_MS` | Optional search deadline; defaults to `6000`. |
| `VOICE_AGENT_HTTP_TIMEOUT_MS` | Optional bridge deadline; defaults to `5000`. |
| `VOICE_AGENT_HTTP_MAX_ATTEMPTS` | Optional bridge attempt limit; defaults to `4`. |
| `VOICE_AGENT_HTTP_RETRY_BASE_MS` | Optional exponential-backoff base; defaults to `250`. |
| `LOG_LEVEL` | Worker logs; keep `fatal` for query privacy. |

Never expose the LiveKit API secret, bridge token, or Tavily key to a browser
client. `.env*` files are ignored except for the credential-free example.

Explicit dispatch metadata must be exactly:

```json
{ "voice_session_id": "<server-created-session-id>" }
```

Unknown fields, missing IDs, and invalid JSON are rejected.

## Internal bridge contract

All requests use `Authorization: Bearer <VOICE_AGENT_BRIDGE_TOKEN>` and accept
any successful 2xx response. Retryable transport failures and retryable status
classes use the configured bounded exponential backoff.

- `GET /internal/voice/sessions/:id/context` returns
  `{ "session": { "id": "…", "status": "pending|active" }, "voice_key": "jacqueline|blake|robyn", "messages": [{ "role": "user|assistant", "content": "…", "created_at": "ISO-8601" }], "next_sequence": 0 }`.
  A missing `voice_key` defaults to `jacqueline` during the backend-first
  rollout. Unknown keys and raw provider voice IDs fail validation. Terminal or
  expired sessions are rejected before the worker joins; `next_sequence` keeps
  worker rejoin writes collision-free.
- `POST /internal/voice/sessions/:id/started` with the server-issued
  `{ "job_id": "AJ_…", "room_id": "RM_…" }`.
- `POST /internal/voice/sessions/:id/turns` with `{ "turns": [turn] }`.
- `POST /internal/voice/sessions/:id/finalize` with
  `{ "turns": [...], "end_reason": "…" }`.
- `POST /internal/voice/sessions/:id/fail` with
  `{ "turns": [...], "reason": "…" }`.

A turn is exactly
`{ item_id, sequence, role, content, created_at, interrupted }`. The backend
owns user/assistant identities and message publication; the worker never accepts
those identities from the browser. Final history is authoritative even after
incremental callback writes.

## Toolchain

Python 3.14 and uv 0.11.28 or newer within uv's 0.11 release line are required.
The tools are complementary:

- uv selects Python, resolves and locks dependencies, and creates the virtual
  environment.
- Ruff formats and lints Python source.
- ty performs static type checking.

ty does not install dependencies and does not format or lint source, so it does
not replace uv or Ruff.

`uv.toml` is the Python equivalent of this worker's former `bunfig.toml`: it
requires the approved uv release line, excludes package artifacts newer than
seven days, refuses source builds and arbitrary PEP 517 build code, and uses
the dependency-confusion-resistant `first-index` strategy. `uv.lock` is
committed for reproducible installs. This is a non-published application, so uv
does not build or install the repository itself as a package.

## Development

```console
uv sync --locked
cp .env.example .env.local
uv run poe download-files
uv run poe check
uv run poe dev
```

Poe exposes the complete local workflow:

| Command | Action |
| --- | --- |
| `uv run poe format` | Format source and tests with Ruff. |
| `uv run poe format-check` | Verify formatting without rewriting files. |
| `uv run poe lint` | Run Ruff lint checks. |
| `uv run poe typecheck` | Type-check the `src` layout with ty. |
| `uv run poe test` | Run deterministic tests and exclude credentialed evals. |
| `uv run poe test-eval` | Run only credential-gated LiveKit evaluations. |
| `uv run poe download-files` | Download installed LiveKit plugin model assets. |
| `uv run poe dev` | Start the worker in LiveKit development mode. |
| `uv run poe start` | Start the production worker process. |
| `uv run poe check` | Run formatting, lint, type, deterministic test, compile, and import gates. |

`poe test` needs no network credentials. `poe test-eval` opts into the three
LiveKit agent-session evaluations, may call LiveKit Inference, and therefore
requires valid LiveKit credentials and intentional model usage.

For an end-to-end audio test, run `uv run poe dev` and use a frontend that
creates an explicit dispatch with the required metadata. The generic Agent
Console has no job metadata, so use a real dispatched room when testing the
bridge.

## Container

The multi-stage Dockerfile uses Debian Bookworm/glibc with Python 3.14 and
pinned uv 0.11.28. It installs only locked runtime wheels, compiles Python
bytecode, downloads LiveKit plugin assets during the build, excludes development
dependencies, and runs as an unprivileged user.

```console
docker build -t realtime-chat-voice-python .
docker run --rm --env-file .env.local realtime-chat-voice-python
```

`no-build = true` is enforced inside the image. A missing compatible wheel must
be solved by selecting a compatible dependency version; do not add compilers or
relax the wheel-only policy.

## LiveKit Cloud deployment

`livekit.toml` is intentionally ignored because it binds a checkout to a
specific LiveKit Cloud agent. After a fresh clone, authenticate and recreate the
local binding for the existing production agent:

```console
lk cloud auth
lk agent config --id {ID} .
```

This binds the directory to `realtime-chat-voice` in the `tessuract` project.
Use `lk agent create` only when intentionally creating a different agent.

Run the quality gate before deploying to the current `ap-south` region:

```console
uv sync --locked
uv run poe check
lk agent deploy --region ap-south --yes .
```

The deployment reuses secrets stored on the LiveKit Cloud agent. Do not put
`LIVEKIT_API_SECRET`, `VOICE_AGENT_BRIDGE_TOKEN`, or `TAVILY_API_KEY` on a
command line or in `livekit.toml`. Wait for a healthy `Running` replica, inspect
`lk agent status` and `lk agent logs`, then smoke-test a new explicitly
dispatched voice session. Repository setup and tests never deploy automatically.
