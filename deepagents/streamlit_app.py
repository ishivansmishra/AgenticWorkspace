from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any, Literal

import streamlit as st
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend, StateBackend, StoreBackend
from dotenv import load_dotenv
from groq import APIStatusError
from langchain_core.messages import BaseMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from pydantic import BaseModel, Field
from tavily import TavilyClient


WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
DEEPAGENTS_ROOT = WORKSPACE_ROOT / "deepagents"
PROJECT_MEMORY_PATH = DEEPAGENTS_ROOT / "project" / "AGENTS.md"
LANGGRAPH_SKILL_PATH = DEEPAGENTS_ROOT / "skills" / "langgraph" / "skill.md"


class ResearchFinding(BaseModel):
    summary: str = Field(description="Summary of findings")
    confidence: float = Field(description="Confidence score from 0 to 1")
    sources: list[str] = Field(description="List of source URLs")


def load_text(path: Path) -> str:
    if path.exists():
        return path.read_text(encoding="utf-8")
    return ""


def as_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for block in value:
            if isinstance(block, dict):
                if "text" in block:
                    parts.append(str(block["text"]))
                elif "content" in block:
                    parts.append(str(block["content"]))
                else:
                    parts.append(json.dumps(block, default=str))
            else:
                parts.append(str(block))
        return "\n".join(parts)
    return str(value)


def message_role_and_content(message: Any) -> tuple[str, str]:
    if isinstance(message, dict):
        role = str(message.get("role", "assistant"))
        content = as_text(message.get("content", ""))
        return role, content

    if isinstance(message, BaseMessage):
        role = getattr(message, "type", message.__class__.__name__.replace("Message", "").lower())
        content = as_text(getattr(message, "content", ""))
        return role, content

    return "assistant", as_text(message)


def build_search_tool(api_key: str):
    client = TavilyClient(api_key=api_key)

    def internet_search(
        query: str,
        max_results: int = 5,
        topic: Literal["general", "news", "finance", "sports"] = "general",
        include_raw_content: bool = False,
    ):
        """Run a Tavily web search for the given query and return search results."""
        return client.search(
            query,
            max_results=max_results,
            topic=topic,
            include_raw_content=include_raw_content,
        )

    return internet_search


def build_backend(selection: str):
    if selection == "StateBackend":
        return StateBackend(), "state"
    if selection == "FilesystemBackend":
        return FilesystemBackend(root_dir=str(WORKSPACE_ROOT), virtual_mode=True), "filesystem"
    if selection == "StoreBackend":
        store = st.session_state.setdefault("store_backend", InMemoryStore())
        return StoreBackend(namespace=lambda _runtime: ("streamlit-chat",), store=store), "store"
    return None, "none"


def build_system_prompt(
    base_prompt: str,
    include_project_memory: bool,
    include_langgraph_skill: bool,
    enable_subagents: bool,
    enable_structured_output: bool,
    backend_label: str,
) -> str:
    sections = [base_prompt.strip()]

    if include_project_memory:
        project_memory = load_text(PROJECT_MEMORY_PATH)
        if project_memory:
            sections.append("Project memory from AGENTS.md:\n" + project_memory)

    if include_langgraph_skill:
        skill_text = load_text(LANGGRAPH_SKILL_PATH)
        if skill_text:
            sections.append("LangGraph skill reference:\n" + skill_text)

    backend_note = {
        "none": "Use normal conversational responses. Do not rely on file persistence.",
        "state": "You have ephemeral file state in memory and can write to /notes/* within the agent state.",
        "filesystem": "You can write files on the local filesystem through the virtual filesystem backend.",
        "store": "Use the store backend for cross-turn persistence by namespace.",
    }[backend_label]
    sections.append(backend_note)

    if enable_subagents:
        sections.append(
            "Use the research subagent for deeper factual investigation, cross-checking, or long research tasks."
        )

    if enable_structured_output:
        sections.append(
            "When the user asks for research output, return a compact JSON object with summary, confidence, and sources."
        )

    return "\n\n".join(section for section in sections if section)


def build_agent(
    model_name: str,
    tools: list[Any],
    system_prompt: str,
    backend: Any | None,
    checkpointer: MemorySaver,
    subagents: list[dict[str, Any]] | None,
):
    kwargs: dict[str, Any] = {
        "model": model_name,
        "system_prompt": system_prompt,
        "checkpointer": checkpointer,
    }
    if tools:
        kwargs["tools"] = tools
    if backend is not None:
        kwargs["backend"] = backend
    if subagents:
        kwargs["subagents"] = subagents
    return create_deep_agent(**kwargs)


def invoke_with_retry(agent, payload: dict[str, Any], config: dict[str, Any], max_retries: int = 3):
    for attempt in range(max_retries):
        try:
            return agent.invoke(payload, config=config)
        except APIStatusError as exc:
            if "rate_limit_exceeded" in str(exc) and attempt < max_retries - 1:
                st.warning(f"Rate limited by Groq. Retrying in 60 seconds (attempt {attempt + 1}/{max_retries}).")
                continue
            raise


def reset_chat_state():
    st.session_state.messages = []
    st.session_state.backend_files = {}


def main():
    load_dotenv()
    st.set_page_config(page_title="Deep Agents Chatbot", page_icon="🤖", layout="wide")

    groq_api_key = os.getenv("GROQ_API_KEY", "")
    tavily_api_key = os.getenv("TAVILY_API_KEY", "")

    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "backend_files" not in st.session_state:
        st.session_state.backend_files = {}
    if "checkpointer" not in st.session_state:
        st.session_state.checkpointer = MemorySaver()

    st.title("Deep Agents Conversational Chatbot")
    st.caption(
        "A Streamlit front end for the deep-agents patterns shown in the notebooks: core agent chat, web search, memory, backends, subagents, and structured research output."
    )

    with st.sidebar:
        st.header("Agent Setup")
        model_name = st.selectbox(
            "Model",
            ["groq:openai/gpt-oss-120b", "groq:openai/gpt-oss-20b"],
            index=0,
        )
        backend_choice = st.radio(
            "Backend",
            ["None", "StateBackend", "FilesystemBackend", "StoreBackend"],
            index=1,
        )
        use_web_search = st.checkbox("Enable Tavily web search", value=True)
        include_project_memory = st.checkbox("Inject AGENTS.md memory", value=True)
        include_langgraph_skill = st.checkbox("Inject langgraph skill context", value=False)
        enable_subagents = st.checkbox("Enable research subagent", value=True)
        enable_structured_output = st.checkbox("Use structured research output prompt", value=True)
        use_checkpoint = st.checkbox("Persist conversation with checkpointer", value=True)
        system_prompt = st.text_area(
            "Base system prompt",
            value=(
                "You are a deep-agents chatbot. Be concise, use tools when helpful, and cite sources when doing research."
            ),
            height=120,
        )
        if st.button("Reset conversation", use_container_width=True):
            reset_chat_state()
            st.rerun()

    if not groq_api_key:
        st.error("GROQ_API_KEY is not set. The chatbot cannot run until the key is available.")
        st.stop()

    tools: list[Any] = []
    search_tool = None
    if use_web_search:
        if not tavily_api_key:
            st.warning("TAVILY_API_KEY is not set, so web search is disabled for this session.")
        else:
            search_tool = build_search_tool(tavily_api_key)
            tools.append(search_tool)

    backend, backend_label = build_backend(backend_choice)
    effective_system_prompt = build_system_prompt(
        base_prompt=system_prompt,
        include_project_memory=include_project_memory,
        include_langgraph_skill=include_langgraph_skill,
        enable_subagents=enable_subagents,
        enable_structured_output=enable_structured_output,
        backend_label=backend_label,
    )

    subagents: list[dict[str, Any]] | None = None
    if enable_subagents and search_tool is not None:
        subagents = [
            {
                "name": "research-agent",
                "description": "Used to research more in-depth questions",
                "system_prompt": "You are a focused research assistant. Use search carefully and return grounded findings.",
                "tools": [search_tool],
                "model": "groq:openai/gpt-oss-20b",
            }
        ]

    thread_id = st.session_state.session_id
    if use_checkpoint:
        config = {"configurable": {"thread_id": thread_id}}
    else:
        config = {}

    agent = build_agent(
        model_name=model_name,
        tools=tools,
        system_prompt=effective_system_prompt,
        backend=backend,
        checkpointer=st.session_state.checkpointer if use_checkpoint else MemorySaver(),
        subagents=subagents,
    )

    left_col, right_col = st.columns([1.35, 1])

    with left_col:
        tab_chat, tab_features = st.tabs(["Chat", "Feature Map"])

        with tab_chat:
            for message in st.session_state.messages:
                role, content = message_role_and_content(message)
                with st.chat_message(role):
                    st.markdown(content)

            prompt = st.chat_input("Ask the deep agent something...")
            if prompt:
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                payload: dict[str, Any] = {"messages": st.session_state.messages}
                if backend_label == "state" and st.session_state.backend_files:
                    payload["files"] = st.session_state.backend_files

                with st.chat_message("assistant"):
                    with st.spinner("Thinking with deep agents..."):
                        result = invoke_with_retry(agent, payload, config=config)

                    last_message = result["messages"][-1]
                    assistant_text = as_text(getattr(last_message, "content", last_message))
                    st.markdown(assistant_text)

                    if enable_structured_output:
                        try:
                            parsed = json.loads(assistant_text)
                            st.success("Structured output parsed successfully.")
                            st.json(parsed)
                        except json.JSONDecodeError:
                            st.info("The response was not valid JSON, so it is shown as normal text.")

                    if result.get("files"):
                        st.session_state.backend_files = result["files"]
                        with st.expander("Agent files"):
                            st.code(json.dumps(result["files"], indent=2, default=str), language="json")

                st.session_state.messages.append({"role": "assistant", "content": assistant_text})

        with tab_features:
            st.subheader("Notebook coverage")
            st.write("This app exposes the deep-agents features demonstrated across the notebooks in this folder.")

            feature_items = [
                "Core agent chat using create_deep_agent with a Groq model.",
                "Optional Tavily web search tool for live research.",
                "Context engineering via the base system prompt and notebook memory text.",
                "AGENTS.md project memory injection.",
                "LangGraph skill context injection from the local skill file.",
                "Backend selection: no backend, StateBackend, FilesystemBackend, or StoreBackend.",
                "Research subagent delegation for deeper investigation.",
                "Structured research output prompt with summary, confidence, and sources.",
            ]
            for item in feature_items:
                st.markdown(f"- {item}")

            st.subheader("Current configuration")
            st.write({
                "model": model_name,
                "backend": backend_choice,
                "web_search": use_web_search and search_tool is not None,
                "subagents": enable_subagents,
                "structured_output": enable_structured_output,
                "checkpoint": use_checkpoint,
                "thread_id": thread_id,
            })

            with st.expander("Project memory preview"):
                st.code(load_text(PROJECT_MEMORY_PATH) or "AGENTS.md not found.", language="markdown")

            with st.expander("LangGraph skill preview"):
                st.code(load_text(LANGGRAPH_SKILL_PATH) or "skill.md not found.", language="markdown")

            with st.expander("Structured schema"):
                st.json(ResearchFinding.model_json_schema())

    with right_col:
        st.subheader("Conversation state")
        st.write(f"Messages: {len(st.session_state.messages)}")
        if st.session_state.backend_files:
            st.write("Backend files detected from the latest response.")
            st.code(json.dumps(st.session_state.backend_files, indent=2, default=str), language="json")
        else:
            st.info("No backend files have been returned yet.")

        st.subheader("How to run")
        st.code("streamlit run deepagents/streamlit_app.py", language="bash")

        st.subheader("Notes")
        st.write(
            "The app uses the same notebook ideas but turns them into one interactive interface. If you want a stricter structured-output mode, ask the agent for research answers and it will be prompted to return JSON."
        )


if __name__ == "__main__":
    main()