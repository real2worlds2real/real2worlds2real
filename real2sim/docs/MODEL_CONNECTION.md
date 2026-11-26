# Model connection and per-user authorization

The preferred interactive entry uses your existing Codex/Claude conversation.
Read `AGENTS.md` and `skills/use-real2sim/SKILL.md`, then use `real2sim-harness task`.
Those domain commands do not start a model, read model credentials or install a
second conversation loop. Native source/media access, subagents, sandbox grants
and conversation history belong to your coding client. Repository-generated
Blender code still uses the shared task execution boundary.

The account and app-server instructions below apply to the optional **managed
session** (`generate`, `resume`, `cousins`). They are not required by the external
task entry. Both entries call the same asset/runtime services and acceptance
contracts; a fresh external task stores context and operation records under
`external/`, without a transport Codex home.

The model is an external, replaceable service. `providers.py` contains the Responses client factories; `codex_transport.py` owns process launch and authorization for the persistent Codex backend. `codex_backend.py` handles the object-generation loop and tool feedback. None contains a user account, access token, refresh token, account ID or pre-authorized home directory.

There are two independent choices:

* **Execution backend**: persistent `codex-app-server` (default), or the more limited `responses` loop. Automatic asset analysis, native deformable tools and cousin orchestration require the persistent backend in this release.
* **Authorization**: `api-key` or `chatgpt-plan`. Always chosen explicitly; neither falls back to the other for billing.

## Your API key

Provide `OPENAI_API_KEY` through your own shell/session or secret manager. Do not put a real key in a source file, command committed to Git, example config or experiment intended for sharing.

```bash
# OPENAI_API_KEY must already be set in this terminal.
real2sim-harness generate --reference /path/to/object.jpg --blender /path/to/blender \
  --auth api-key --model gpt-6-astra
```

`OPENAI_BASE_URL` optionally selects a compatible endpoint for API-key mode; it applies to the persistent transport and to the SDK client. Endpoint compatibility with Responses, tools, images and the chosen model is the provider's responsibility. ChatGPT-plan mode always uses the official endpoint.

## Your ChatGPT account

Install the `chatgpt` extra and create a private authorization directory. It may live outside the checkout. The helper validates owner-only directory/file permissions and uses explicit browser authorization:

```bash
real2sim-harness-auth login --auth-dir /path/to/your/private/real2sim-harness-auth
real2sim-harness-auth status --auth-dir /path/to/your/private/real2sim-harness-auth
real2sim-harness-auth models --auth-dir /path/to/your/private/real2sim-harness-auth
real2sim-harness generate --reference /path/to/object.jpg --blender /path/to/blender \
  --auth chatgpt-plan --auth-dir /path/to/your/private/real2sim-harness-auth --model gpt-6-astra
```

The login helper exposes a loopback browser flow; open the local URL that it prints. On a remote machine use your own loopback forwarding setup. No shared host, account or tunnel destination is prescribed. Plan access requires the relevant account permission and usage authorization. The tool does not promise that every account can use every model.

To switch accounts, use a different private directory and authorize that account there, then pass its directory explicitly. Each collaborator should authorize separately. Do not distribute an existing `credentials.json`, `host.json`, session lock, Codex home or run folder. An expired authorization can be renewed from that user's own stored grant; a missing authorization fails rather than borrowing credentials from another account.

The worker uses an isolated Codex home within the ignored run directory. It does not read personal Codex login files, inherited MCP servers or project instructions. The host supplies only the selected credential to the worker process environment; the secret is absent from model context, command arguments and tracked configuration. Per-run and stderr diagnostics redact that credential.

## Source and task access

The optional complete persistent backend supports **both API-key and ChatGPT-plan
authorization** with the same task access. Authorization chooses whose model
service is used; it does not change what local evidence the worker can inspect.
API-key use does not require switching to the early `responses` backend.

`project_inspection.py` generates paths from the actual checkout and current run.
The worker's `project_access.json` advertises readable framework source, the full
current task record root, the writable agent workspace and its analysis Python
environment. The worker can search/read source, inspect logs/configuration and
retained numerical arrays, import framework modules, and write/run its own
analysis in the workspace. This is on-demand investigation; the harness does not
have to preselect every log or invent one tool for every defect.

Common working methods and the selected task workflow are included in developer
instructions at startup and resume. `codex/startup_skills.json` records their names
and content hashes. Specialized skills and their detailed resources remain
on-demand. The common skill routes video input and candidate-movie review to
`inspect-video`, including the limits of sparse frames, thumbnails and partial
motion coverage. It does not impose a fixed inspection recipe or certify that
all visual defects will be found. The limited direct Responses loop also receives
the startup skill text, but retains its smaller tool set described below.

External tasks retain the same layers in `external/startup_skills.json`, plus a
fingerprint of current capability descriptions. On later domain calls the
framework checks for changed common/task guidance or capability descriptions,
refreshes context without replacing authored source or prior attempts, and
returns the current instructions in `guidance_update`. If that call requested
acceptance, it is rejected before dispatch; apply the update, recheck affected
findings and retry through the normal acceptance gate. Other calls retain their
ordinary result with the update attached. This delivers current guidance; it
does not prove that an external agent followed it or inspected every defect.

| Location | Codex native file/command access |
| --- | --- |
| Selected checkout's public source/docs/scripts/skills/tests and related source entries | Read only |
| This task's retained inputs, artifacts, parameters, logs and simulator states | Read only |
| This task's agent workspace, analysis home, temporary files and caches | Read/write |
| Transport Codex home and selected authorization directory | Denied |
| Other personal files and unrelated tasks | No grant |

Normal system/Python runtime dependencies remain readable. The profile starts
from the built-in read-only profile, denies broad filesystem access, and grants
the selected source/task/workspace paths explicitly. Analysis temporary files
and caches live inside the task workspace. Both new and resumed conversations
explicitly select the task profile; `codex/permissions_receipt.json` records the
profile reported by the local server. A compatible Codex CLI must enforce these
permissions; unsupported profiles are not a reason to retry with full access.

Generated Blender code uses a separate shared Linux execution boundary through
`task_execution.py`. Both the persistent Codex backend and the direct Responses
prototype bind the Blender executor to the current task; the same bound executable
is used for exports, previews and motion replay. Source and task evidence remain
readable. Only declared task output directories, workspace and task-local
temporary/cache directories are writable. The selected authorization store and
Codex transport home are masked; inherited model/other service tokens are not
passed into the tool environment. Child processes remain inside that boundary.
Generated parametric fitting code uses the same process boundary with its retained
inputs and output directory.

Linux `bubblewrap` must be installed and permitted by the host. Missing support
fails the tool instead of silently falling back to unrestricted execution.
Installed runtime dependencies are read-only; additional operator-owned dependency
locations can be declared with `REAL2SIM_TOOL_READ_PATHS` as a JSON list of paths.
Keep it limited to the dependencies actually needed. Tool execution does not grant
network access. Operator-configured simulator/training runner commands remain
trusted integration code and must supply their own compatible execution setup;
this is a task boundary, not a general security certification for arbitrary
external adapters. Do not store credentials or unrelated personal files in a task.

The early direct `--backend responses` loop receives only the content and tools
explicitly supplied by `agent.py`; it does not inherit Codex's native filesystem,
shell, skill and full asset/scene tool capabilities. It is a limited rigid/object
prototype, not a second equivalent full harness. See the capability guards in
`cli.py`. Do not describe API-key authorization and this legacy backend as the
same choice, or claim that adding an API key automatically grants local access.

## Replace the connection

* Change only the account: pass a different `--auth-dir`, or a different API key through the process environment.
* Change the model: set `--model` to an ID available to that authorization. No account identity is encoded in a model ID.
* Change a compatible API provider: set `OPENAI_BASE_URL` in API-key mode and validate its tool/image support.
* Integrate another transport: implement the client/launch boundary without altering the Blender tools or asset semantics. Retain explicit authorization, evidence routing and acceptance checks.

The existing connection code uses the [official Codex app-server token-sharing route](https://developers.openai.com/siwc/token-sharing-open-source/codex-app-server) and [explicit sign-in flow](https://developers.openai.com/siwc/token-sharing-open-source/sign-in). Availability and permissions follow the user's account and the current official service; source-only tests do not log in or consume model usage.
