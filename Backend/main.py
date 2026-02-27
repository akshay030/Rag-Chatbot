from __future__ import annotations

import os
import json
import requests
import tempfile
import time
from typing import Annotated, Dict, Any, Optional, TypedDict

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# =========================
# LangChain / LangGraph
# =========================
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.tools import DuckDuckGoSearchRun

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage,
)

from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver

from groq import RateLimitError

# =========================
# INIT
# =========================
load_dotenv()
checkpointer = InMemorySaver()

# =========================
# FASTAPI
# =========================
app = FastAPI(title="RAG Agentic Chatbot (Stable Version)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# LLM
# =========================
llm = ChatGroq(
    model="qwen/qwen3-32b",
    temperature=0,
    max_tokens=400,   # 🔥 reduce token explosion
)

# =========================
# EMBEDDINGS
# =========================
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# =========================
# RAG STORAGE
# =========================
_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, dict] = {}

def _get_retriever(thread_id: Optional[str]):
    if not thread_id:
        return None

    thread_id = str(thread_id)

    # Already in memory
    if thread_id in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[thread_id]

    # Try loading from disk
    user_dir = os.path.join("vector_store", thread_id)

    if os.path.exists(user_dir):
        store = FAISS.load_local(
            user_dir,
            embeddings,
            allow_dangerous_deserialization=True,
        )
        retriever = store.as_retriever(search_kwargs={"k": 2})
        _THREAD_RETRIEVERS[thread_id] = retriever
        return retriever

    return None

# =========================
# TOOLS
# =========================
search = DuckDuckGoSearchRun(region="us-en")

def calculator_tool(a: float, b: float, op: str) -> dict:
    if op == "add":
        return {"result": a + b}
    if op == "sub":
        return {"result": a - b}
    if op == "mul":
        return {"result": a * b}
    if op == "div":
        return {"result": a / b if b != 0 else "division by zero"}
    return {"error": "invalid operation"}

def web_search_tool(query: str) -> dict:
    return {"result": search.run(query)}

def rag_tool(query: str, thread_id: Optional[str] = None) -> dict:
    retriever = _get_retriever(thread_id)

    if retriever is None:
        return {
            "error": "No document indexed. Upload a PDF first.",
            "query": query,
        }

    docs = retriever.invoke(query)

    formatted_context = "\n\n".join(
        [f"Source {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)]
    )

    return {
        "query": query,
        "context": formatted_context,
        "source_file": _THREAD_METADATA.get(str(thread_id), {}).get("filename"),
    }

def stock_price_tool(symbol: str) -> dict:
    try:
        resp = requests.post(
            "https://mcp.alphavantage.co/",
            json={
                "tool": "stock_price",
                "arguments": {"symbol": symbol},
            },
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"error": str(e)}

# =========================
# BIND TOOLS
# =========================
llm = llm.bind_tools([
    calculator_tool,
    web_search_tool,
    rag_tool,
    stock_price_tool,
])

TOOLS = {
    "calculator_tool": calculator_tool,
    "web_search_tool": web_search_tool,
    "rag_tool": rag_tool,
    "stock_price_tool": stock_price_tool,
}

# =========================
# SAFE LLM CALL (429 FIX)
# =========================
def safe_llm_invoke(messages):
    for _ in range(3):
        try:
            return llm.invoke(messages)
        except RateLimitError:
            time.sleep(25)
    raise Exception("LLM rate limit exceeded")

# =========================
# LANGGRAPH STATE
# =========================
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# =========================
# TOOL NODE
# =========================
def run_tools(state: ChatState, config=None):
    last_msg = state["messages"][-1]

    if not isinstance(last_msg, AIMessage):
        return {}

    if not last_msg.tool_calls:
        return {}

    thread_id = config["configurable"]["thread_id"]
    outputs = []

    for call in last_msg.tool_calls:
        name = call["name"]
        args = call["args"]

        if name == "rag_tool":
            args["thread_id"] = thread_id

        if name in TOOLS:
            result = TOOLS[name](**args)
            outputs.append(
                ToolMessage(
                    content=json.dumps(result),
                    tool_call_id=call["id"],
                )
            )

    return {"messages": outputs}

# =========================
# CHAT NODE
# =========================
def chat_node(state: ChatState, config=None):
    thread_id = config["configurable"]["thread_id"]

    system_prompt = SystemMessage(
        content=f"""
You are a professional AI assistant.

Rules:
- Never mention tools or APIs.
- Use uploaded documents when relevant.
- Only call tools when necessary.
- Answer clearly and concisely.

Thread ID: {thread_id}
"""
    )

    messages = state["messages"]

    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [system_prompt, *messages]

    response = safe_llm_invoke(messages)
    return {"messages": [response]}

# =========================
# GRAPH
# =========================
graph = StateGraph(ChatState)

graph.add_node("chat", chat_node)
graph.add_node("tools", run_tools)

graph.add_edge(START, "chat")

graph.add_conditional_edges(
    "chat",
    lambda s: "tools"
    if isinstance(s["messages"][-1], AIMessage)
    and s["messages"][-1].tool_calls
    else "__end__"
)

graph.add_edge("tools", "chat")

chatbot = graph.compile(checkpointer=checkpointer)

# =========================
# API MODELS
# =========================
class ChatRequest(BaseModel):
    user_id: str
    message: str

# =========================
# ENDPOINTS
# =========================
@app.post("/chat")
def chat(req: ChatRequest):
    result = chatbot.invoke(
        {"messages": [HumanMessage(content=req.message)]},
        config={"configurable": {"thread_id": req.user_id}},
    )
    return {"reply": result["messages"][-1].content}

@app.post("/upload")
@app.post("/upload")
def upload_pdf(
    user_id: str = Form(...),
    file: UploadFile = File(...),
):
    VECTOR_BASE_DIR = "vector_store"
    os.makedirs(VECTOR_BASE_DIR, exist_ok=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(file.file.read())
        temp_path = f.name

    try:
        # Load PDF
        loader = PyMuPDFLoader(temp_path)
        docs = loader.load()

        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=700,
            chunk_overlap=100,
        )
        chunks = splitter.split_documents(docs)

        # Create FAISS store
        store = FAISS.from_documents(chunks, embeddings)

        # 🔥 Create user directory
        user_dir = os.path.join(VECTOR_BASE_DIR, str(user_id))
        os.makedirs(user_dir, exist_ok=True)

        # 🔥 Save FAISS to disk
        store.save_local(user_dir)

        # Create retriever
        retriever = store.as_retriever(search_kwargs={"k": 2})

        # Cache in memory
        _THREAD_RETRIEVERS[str(user_id)] = retriever
        _THREAD_METADATA[str(user_id)] = {
            "filename": file.filename,
            "documents": len(docs),
            "chunks": len(chunks),
        }

        return {
            "message": "PDF uploaded and indexed successfully",
            **_THREAD_METADATA[str(user_id)],
        }

    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass