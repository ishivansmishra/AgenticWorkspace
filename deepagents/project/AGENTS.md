# Deep Agents — Project Memory

Persistent context for agents in this workspace. Loaded automatically when passed to `create_deep_agent` via the `memory` parameter.

## Role

You are a Deep Agent operating inside the **AgenticWorkspace** `deepagents/` learning and experimentation project. You help with research, coding, file operations, and multi-step tasks using the LangChain Deep Agents harness.

## Conventions

- Load environment variables from the workspace root `.env` (`GROQ_API_KEY`, `TAVILY_API_KEY`).
- Default model: `groq:openai/gpt-oss-120b`.
- Use Tavily (`web_search`) for live web research when tools are available.
- Prefer the virtual filesystem paths (`/notes/...`, `/project/...`) when using a backend; do not assume host OS paths unless `FilesystemBackend` is configured.
- Keep responses concise; cite sources when doing research.
- Use subagents (`task` tool) for parallel or isolated long-running work; return synthesized results to the user.
- Use `write_todos` for multi-step tasks with three or more steps.

## Project Layout

```
deepagents/
├── skills/                # On-demand skill packs for deep agents
│   ├── python/
│   ├── langgraph/
│   ├── aws/
│   └── reporting/
├── project/
│   └── AGENTS.md          # This file — persistent agent memory
├── notes/                 # Agent-written notes (virtual or on disk)
├── basicsdeepagents.ipynb # Intro: simple agent vs deep agent
├── backend.ipynb          # StateBackend, FilesystemBackend, StoreBackend
└── contextengineering.ipynb # System prompt, memory, skills
```

## Loading This File

Pass the path through `memory` when creating an agent. With `FilesystemBackend`, use a virtual path relative to the backend root:

```python
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend

ROOT = "deepagents"  # or "." if running from workspace root

agent = create_deep_agent(
    model="groq:openai/gpt-oss-120b",
    memory=["/project/AGENTS.md"],
    backend=FilesystemBackend(root_dir=ROOT, virtual_mode=True),
)
```

With `StateBackend`, memory paths are resolved against in-memory state. Pre-seed `files` in invoke state or write the file via the agent first.

Memory is **always injected** into the system prompt (unlike skills, which load on demand). Keep this file focused on stable guidelines—not task-specific details.

---

## Deep Agents Architecture

Deep Agents is an agent **harness** built on LangChain and LangGraph. It wraps the standard tool-calling loop with planning, filesystem, subagents, and context management.

### High-Level Flow

```
User message
    → System prompt (custom + base + memory + skills + tool docs)
    → Agent loop (LLM ↔ tools)
        ├── Built-in: write_todos, read/write/edit files, task (subagents)
        ├── Custom tools (e.g. web_search)
        └── Subagents: isolated context → single final report back
    → Response (+ optional files/state updates)
```

### Core Components

| Layer | Purpose | Key APIs / Tools |
|-------|---------|------------------|
| **Execution** | Act in an environment | Custom `tools`, virtual FS (`ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`) |
| **Context** | What the agent knows | `system_prompt`, `memory`, `skills`, summarization & offloading |
| **Delegation** | Break down work | `write_todos`, `task` (subagents) |
| **Steering** | Human control | `interrupt_on` for approval before sensitive tools |
| **Persistence** | Cross-thread state | Backends + optional LangGraph `store` / checkpointer |

### Backends (Virtual Filesystem)

| Backend | Storage | Use Case |
|---------|---------|----------|
| `StateBackend()` | In graph state (`files` key) | Ephemeral files per thread; pass `files` on follow-up invokes |
| `FilesystemBackend(root_dir, virtual_mode=True)` | Real disk under `root_dir` | Persistent notes, project files on local filesystem |
| `StoreBackend(namespace=...)` | LangGraph store | Cross-conversation persistence with namespaces |

Default backend is in-memory state if none is specified.

### Context Engineering (System Prompt Order)

When assembled, the system prompt typically includes (in order):

1. Custom `system_prompt` (if provided)
2. Base agent instructions
3. To-do list / planning guidance
4. **Memory** — content from `AGENTS.md` files (this document)
5. **Skills** — on-demand workflows from `SKILL.md` directories
6. Virtual filesystem & execute tool documentation
7. Subagent / `task` tool usage
8. Custom middleware prompts
9. Human-in-the-loop instructions (if `interrupt_on` is set)

### Subagents

The `task` tool spawns ephemeral child agents with fresh context. Each subagent runs autonomously and returns one final message. Use for:

- Parallel research on independent topics
- Heavy subtasks that would bloat the main context
- Specialized work when custom `subagents` are configured

Default subagent type: `general-purpose`.

### Built-in vs Plain LangChain Agent

| | `create_agent` | `create_deep_agent` |
|---|----------------|---------------------|
| Planning | Manual | `write_todos` |
| Files | None | Virtual FS + backends |
| Delegation | None | `task` subagents |
| Memory | Manual prompt | `memory=["/project/AGENTS.md"]` |
| Long runs | Limited | Summarization, offloading, store backends |

### Dependencies (This Workspace)

- `deepagents==0.6.12`
- `langchain`, `langgraph`, `langchain-groq`, `langchain-tavily`
- Python 3.14 venv at workspace root

---

## Quick Reference — Minimal Agent

```python
import os
from dotenv import load_dotenv
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend

load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

agent = create_deep_agent(
    model="groq:openai/gpt-oss-120b",
    system_prompt="You are a research assistant. Cite sources.",
    memory=["/project/AGENTS.md"],
    backend=FilesystemBackend(root_dir="deepagents", virtual_mode=True),
)

result = agent.invoke({"messages": [{"role": "user", "content": "..."}]})
```
