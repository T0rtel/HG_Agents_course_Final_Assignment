import subprocess, sys, tempfile, os
import re
import pandas as pd

from dotenv import load_dotenv
from langfuse import get_client
from langfuse.langchain import CallbackHandler
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import START, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama

import json
from langchain_core.documents import Document
from langchain_community.document_loaders import NotebookLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from agent_tools.math_tools import (
    add,
    divide,
    modulus,
    multiply,
    power,
    square_root,
    subtract,
)

from agent_tools.arxiv_search import arxiv_search
from agent_tools.web_search import web_search, web_fetch
from agent_tools.wiki_search import (
    wikipedia_history,
    wikipedia_lookup,
    wikipedia_search,
)


load_dotenv()

langfuse = None
if os.getenv("LANGFUSE_ENABLED", "1").casefold() in {"1", "true", "yes"}:
    if os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
        langfuse = get_client()


def get_langfuse_config(question: str) -> dict:
    """Create the callback configuration for one agent evaluation run."""
    if langfuse is None:
        return {}

    return {
        "callbacks": [CallbackHandler()],
        "run_name": "langgraph-evaluation-agent",
        "metadata": {
            "agent": "langgraph-evaluation-agent",
            "question_length": len(question),
        },
        "tags": ["evaluation", "langgraph"],
    }


def invoke_graph(graph, question: str):
    """Invoke a graph with Langfuse tracing when it is configured."""
    try:
        return graph.invoke(
            {"messages": [HumanMessage(content=question)]},
            config=get_langfuse_config(question),
        )
    finally:
        if langfuse is not None:
            langfuse.flush()


### =============== CODE INTERPRETER TOOLS =============== ###


@tool
def execute_code(code: str) -> str:
    """Execute Python code and return what it prints (stdout) or the error (stderr).
    Always use print() to show the result you want to see."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(code)
        path = f.name

    try:
        result = subprocess.run(
            [sys.executable, path],
            capture_output=True,
            text=True,
            timeout=30,          # stop infinite loops
        )
        output = result.stdout
        if result.stderr:
            output += "\nERROR:\n" + result.stderr
        return output[:5000] or "(no output, did you forget to print()?)"
    except subprocess.TimeoutExpired:
        return "ERROR: code timed out after 30 seconds"
    finally:
        os.remove(path)

# @tool
# def similar_examples(question: str) -> str:
#     """Find similar past questions and how they were solved (steps and tools used).
#     Use this at the start of a task to get ideas for an approach."""
#     results = retriever.invoke(question)
#     return "\n\n---\n\n".join(
#         f"Similar question: {d.page_content}\n"
#         f"Tools used: {d.metadata['tools']}\n"
#         f"Steps taken:\n{d.metadata['steps']}"
#         for d in results
#     )

# load the system prompt from the file
from pathlib import Path

BASE_DIR = Path(__file__).parent
prompt_path = BASE_DIR / "systemprompt.txt"

with open(prompt_path, "r", encoding="utf-8") as f:
    system_prompt = f.read()
print(system_prompt)


def needs_lookup(row_or_question: dict) -> bool:
    q = row_or_question["Question"]
    has_file = bool(row_or_question.get("file_name"))
    has_video = bool(re.search(r"youtube\.com|youtu\.be", q))
    return has_file or has_video

docs = []
with open("metadata.jsonl", encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)
        ann = row.get("Annotator Metadata", {})
        # if needs_lookup(row):
        #             docs.append(Document(
        #                 page_content=row["Question"],
        #                 metadata={
        #                     "task_id": row["task_id"], 
        #                     "answer": row["Final answer"]},
        #             ))
        # else:
        docs.append(Document(
                        page_content=row["Question"],   # only this gets embedded
                        metadata={
                            "task_id": row["task_id"],
                            "level": row.get("Level"),
                            "steps": ann.get("Steps", ""),
                            "tools": ann.get("Tools", ""),
                            #"answer": row["Final answer"],
                            # deliberately NOT storing "Final answer"
                        },
                    ))


nb_docs = NotebookLoader(
    "explore_metadata.ipynb", include_outputs=True, max_output_length=20, remove_newline=True
).load()
nb_chunks = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100).split_documents(nb_docs)
docs += nb_chunks

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(docs, embeddings, persist_directory="./chroma_db")
retriever_store = vectorstore.as_retriever(search_kwargs={"k": 3})

tools = [
    web_fetch,
    web_search,
    arxiv_search,
    wikipedia_search,
    wikipedia_lookup,
    wikipedia_history,
    multiply,
    add,
    subtract,
    divide,
    modulus,
    power,
    square_root,
    execute_code,
    #similar_examples,
]


# Build graph function
def build_graph(provider: str = "groq"):
    """Build the graph"""
    # Load environment variables from .env file
    if provider == "groq":
        # Groq https://console.groq.com/docs/models
        llm = ChatGroq(
                    model= os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b"),
                    groq_api_key=os.environ["GROQ_API_KEY"],
                    temperature=0.1
                )
        
    elif provider == "google":
        llm = ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-3.8-flash"),
        google_api_key=  os.getenv("GEMINI_API_KEY"),
        temperature=0.1,
        max_output_tokens=512,
        max_retries=0,
        )

    elif provider == "ollama":
        llm = ChatOllama(
        model=os.getenv("OLLAMA_MODEL", "qwen3:14b"),
        temperature=0.1,
        top_p=0.95,
        num_ctx=16384,
        reasoning=True,
        keep_alive="30m",
    )
    else:
        raise ValueError("Invalid provider.")
    # Bind tools to LLM
    llm_with_tools = llm.bind_tools(tools)

    # Node
    def assistant(state: MessagesState):
        """Assistant node"""
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    def retriever_func(state: MessagesState):
        """Retriever node"""
        similar_questions = retriever_store.invoke(
            state["messages"][-1].content
        )

        if similar_questions:
            examples = "\n\n---\n\n".join(
                document.page_content for document in similar_questions
            )
            example_msg = SystemMessage(
                content=(
                    "Relevant past questions and their research guides follow. "
                    "Use them as examples, verify facts with tools when needed, "
                    "and do not copy an answer unless it answers the current question.\n\n"
                    f"{examples}"
                ),
            )
            return {"messages": [SystemMessage(content=system_prompt)] + state["messages"] + [example_msg]}
        else:
            return {"messages": [SystemMessage(content=system_prompt)] + state["messages"]}

    builder = StateGraph(MessagesState)
    builder.add_node("retriever", retriever_func)
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "retriever")
    builder.add_edge("retriever", "assistant")
    builder.add_conditional_edges(
        "assistant",
        tools_condition,
    )
    builder.add_edge("tools", "assistant")

    # Compile graph
    return builder.compile()


# test
if __name__ == "__main__":
    question = "When was a picture of St. Thomas Aquinas first added to the Wikipedia page on the Principle of double effect?"
    graph = build_graph(provider="ollama")
    messages = invoke_graph(graph, question)
    for m in messages["messages"]:
        m.pretty_print()
