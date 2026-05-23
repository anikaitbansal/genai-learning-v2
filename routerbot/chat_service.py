import logging
from typing import cast
from chat_log_repository import ChatLogRepository
from langgraph_flow import build_langgraph_flow, GraphState

logger = logging.getLogger(__name__)


class ChatService:
    def __init__(self, memory, retriever, debug=False):
        self.memory = memory
        self.debug = debug
        self.retriever = retriever
        self.chat_log_repository = ChatLogRepository()
        self.graph = build_langgraph_flow()

    def process_message(
        self, message, session_id, request_id, user_id, use_rag=True, debug=False
    ):
        cleaned_message = message.strip()

        if not cleaned_message:
            raise ValueError("Input message cannot be empty or whitespace.")

        logger.info(
            "[request_id=%s] flow=start session_id=%s message_length=%s use_rag=%s debug=%s",
            request_id,
            session_id,
            len(cleaned_message),
            use_rag,
            debug or self.debug,
        )

        logger.info(
            f"[request_id={request_id}] flow=graph_start session_id={session_id}"
        )

        initial_state = {
            "original_message": cleaned_message,
            "session_id": session_id,
            "user_id": user_id,
            "use_rag": use_rag,
            "retriever": self.retriever,
            "intent": "",
            "retrieved_chunks": [],
            "rag_used": False,
            "bot_reply": "",
            "evaluation": {},
            "evaluation_reason": "",
            "retry_count": 0,
        }

        final_state = self.graph.invoke(cast(GraphState, initial_state))

        intent = final_state["intent"]
        retrieved_chunks = final_state["retrieved_chunks"]
        rag_used = final_state["rag_used"]
        response = final_state["bot_reply"]
        evaluation_result = final_state["evaluation"]
        retry_count = final_state["retry_count"]
        retry_happened = retry_count > 0

        logger.info(
            f"[request_id={request_id}] flow=graph_done session_id={session_id} intent={intent} rag_used={rag_used} retry_count={retry_count} response_length={len(response)}"
        )

        logger.info(
            "[request_id=%s] flow=db_save_start session_id=%s intent=%s rag_used=%s",
            request_id,
            session_id,
            intent,
            rag_used,
        )

        self.chat_log_repository.save_chat_log(
            session_id=session_id,
            request_id=request_id,
            user_input=cleaned_message,
            bot_reply=response,
            intent=intent,
            rag_used=rag_used,
            user_id=user_id,
        )

        logger.info(
            "[request_id=%s] flow=db_save_done session_id=%s intent=%s rag_used=%s",
            request_id,
            session_id,
            intent,
            rag_used,
        )

        logger.info(
            "[request_id=%s] flow=complete session_id=%s intent=%s rag_used=%s response_length=%s",
            request_id,
            session_id,
            intent,
            rag_used,
            len(response),
        )

        result = {
            "user_message": cleaned_message,
            "bot_reply": response,
            "intent": intent,
            "session_id": session_id,
            "rag_used": rag_used,
        }

        if debug or self.debug:
            result["retrieved_chunks"] = retrieved_chunks
            result["evaluation"] = evaluation_result
            result["retry_count"] = retry_count
            result["retry_happened"] = retry_happened

        return result
