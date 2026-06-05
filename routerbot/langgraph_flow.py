import re
import logging
from uuid import UUID
from typing import TypedDict, Any
from langgraph.graph import StateGraph, END

from retriever import retrieve_as_dicts
from routing import classify_intent, intent_branch
from response_evaluator import ResponseEvaluator
from langchain_memory_adapter import LangChainMemoryAdapter
from config import RAG_TOP_K

logger = logging.getLogger(__name__)


class GraphState(TypedDict):
    original_message: str
    session_id: str
    user_id: UUID
    use_rag: bool
    retriever: Any
    intent: str
    retrieved_chunks: list[dict[str, Any]]
    rag_used: bool
    bot_reply: str
    evaluation: dict[str, str]
    evaluation_reason: str
    retry_count: int


def classify_node(state: GraphState) -> GraphState:
    logger.info(
        "graph_node=classify_start message_length=%s", len(state["original_message"])
    )

    intent = classify_intent(state["original_message"])
    state["intent"] = intent

    logger.info("graph_node=classify_done intent=%s", intent)
    return state


def build_retrieval_query(user_message: str) -> str:
    cleaned = user_message.strip().lower()

    cleaned = re.sub(r"\b(hi|hello|hey|thanks|please|buddy)\b", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    if "name" in cleaned:
        return "person name full name candidate resume cv"

    if "phone" in cleaned or "number" in cleaned:
        return "phone number contact mobile resume"

    if "email" in cleaned:
        return "email address contact resume"

    if "link" in cleaned or "linkedin" in cleaned or "github" in cleaned:
        return "links linkedin github portfolio resume"

    if "skills" in cleaned:
        return "technical skills languages tools resume"

    if "experience" in cleaned or "work" in cleaned:
        return "work experience company role resume"

    if "summary" in cleaned or "about" in cleaned:
        return "summary profile description resume"

    return cleaned


def retrieve_node(state: GraphState) -> GraphState:
    logger.info("graph_node=retrieve_start use_rag=%s", state["use_rag"])

    retrieved_chunks = []
    rag_used = False

    if state["use_rag"]:
        retrieval_query = state["original_message"]
        logger.info(
            "graph_node=retrieve_query_built original_message=%s retrieval_query=%s",
            state["original_message"],
            retrieval_query,
        )

        retrieved_chunks = retrieve_as_dicts(
            state["retriever"], retrieval_query, top_k=RAG_TOP_K
        )

        rag_used = len(retrieved_chunks) > 0

    state["retrieved_chunks"] = retrieved_chunks
    state["rag_used"] = rag_used

    logger.info(
        "graph_node=retrieve_done rag_used=%s retrieved_count=%s",
        rag_used,
        len(retrieved_chunks),
    )
    return state


def generate_node(state: GraphState) -> GraphState:
    logger.info(
        f"graph_node = generate_start intent: {state['intent']} rag_used: {state['rag_used']} retry_count: {state['retry_count']}"
    )

    inputs = {
        "user_input": state["original_message"],
        "session_id": state["session_id"],
        "user_id": state["user_id"],
        "retrieved_chunks": state["retrieved_chunks"],
        "retry_reason": state.get("evaluation_reason", ""),
        "retry_count": state["retry_count"],
    }

    bot_response = intent_branch.invoke(
        {
            "intent": state["intent"],
            "inputs": inputs,
        }
    )

    state["bot_reply"] = bot_response

    logger.info(
        f"graph_node = generate_done response_length: {len(bot_response)} retry_count: {state['retry_count']}"
    )
    return state


def evaluate_node(state: GraphState) -> GraphState:
    logger.info("graph_node=evaluate_start retry_count=%s", state["retry_count"])

    evaluator = ResponseEvaluator()

    adapter = LangChainMemoryAdapter(
        session_id=state["session_id"], user_id=state["user_id"]
    )
    history_messages = adapter.messages

    chat_history_for_eval = []
    for message in history_messages:
        role_map = {"system": "system", "human": "user", "ai": "assistant"}
        role = role_map.get(message.type, "user")
        chat_history_for_eval.append({"role": role, "content": message.content})

    evaluation_result = evaluator.evaluate(
        user_input=state["original_message"],
        bot_response=state["bot_reply"],
        chat_history=chat_history_for_eval,
    )

    state["evaluation"] = evaluation_result
    state["evaluation_reason"] = evaluation_result.get("reason", "").strip()

    logger.info(
        "graph_node=evaluate_done score=%s reason=%s retry_count=%s",
        evaluation_result["score"],
        state["evaluation_reason"],
        state["retry_count"],
    )
    return state


def prepare_retry_node(state: GraphState) -> GraphState:
    state["retry_count"] += 1

    logger.info(
        "graph_node=prepare_retry retry_count=%s reason=%s",
        state["retry_count"],
        state.get("evaluation_reason", ""),
    )
    return state


def route_after_evaluation(state: GraphState) -> str:
    score = state["evaluation"].get("score", "incorrect")

    if score == "correct":
        logger.info(
            "graph_route=finish reason=score_correct retry_count=%s",
            state["retry_count"],
        )
        return "end"

    if state["retry_count"] >= 1:
        logger.info(
            "graph_route=finish reason=retry_limit_reached score=%s retry_count=%s",
            score,
            state["retry_count"],
        )
        return "end"

    logger.info(
        "graph_route=retry reason=score_%s retry_count=%s", score, state["retry_count"]
    )
    return "prepare_retry"


def build_langgraph_flow():
    graph_builder = StateGraph(GraphState)

    graph_builder.add_node("classify", classify_node)
    graph_builder.add_node("retrieve", retrieve_node)
    graph_builder.add_node("generate", generate_node)
    graph_builder.add_node("evaluate", evaluate_node)
    graph_builder.add_node("prepare_retry", prepare_retry_node)

    graph_builder.set_entry_point("classify")

    graph_builder.add_edge("classify", "retrieve")
    graph_builder.add_edge("retrieve", "generate")
    graph_builder.add_edge("generate", "evaluate")
    graph_builder.add_edge("prepare_retry", "generate")

    graph_builder.add_conditional_edges(
        "evaluate",
        route_after_evaluation,
        {"prepare_retry": "prepare_retry", "end": END},
    )

    return graph_builder.compile()
