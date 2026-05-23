import logging
from uuid import UUID
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from memory_manager import MemoryManager

logger = logging.getLogger(__name__)


def dict_to_message(message_dict: dict) -> BaseMessage:
    role = message_dict.get("role", "")
    content = message_dict.get("content", "")

    if role == "system":
        return SystemMessage(content=content)
    if role == "user":
        return HumanMessage(content=content)
    if role == "assistant":
        return AIMessage(content=content)

    logger.warning(f"adapter_stage = unknown_role_defaulted_to_human role: {role}")
    return HumanMessage(content=content)


def message_to_dict(message: BaseMessage) -> dict:
    if isinstance(message, SystemMessage):
        return {"role": "system", "content": message.content}
    if isinstance(message, HumanMessage):
        return {"role": "user", "content": message.content}
    if isinstance(message, AIMessage):
        return {"role": "assistant", "content": message.content}

    logger.warning(
        f"adapter_stage = unknown_message_type_defaulted_to_user type: {type(message).__name__}"
    )
    return {"role": "user", "content": str(message.content)}


class LangChainMemoryAdapter(BaseChatMessageHistory):
    def __init__(self, session_id: str, user_id: UUID):
        self.session_id = session_id
        self.user_id = user_id
        self.memory_manager = MemoryManager(session_id, user_id)

    @property
    def messages(self) -> list[BaseMessage]:
        raw_history = self.memory_manager.load()
        converted = [dict_to_message(item) for item in raw_history]

        logger.info(
            f"adapter_stage = messages_loaded session_id: {self.session_id} "
            f"user_id: {self.user_id} message_count: {len(converted)}"
        )
        return converted

    def add_messages(self, messages: list[BaseMessage]) -> None:
        if not messages:
            return

        current_history = self.memory_manager.load()

        for message in messages:
            current_history.append(message_to_dict(message))

        if len(current_history) > 21:
            current_history = [current_history[0]] + current_history[-20:]

        self.memory_manager.save(current_history)

        added_roles = [message_to_dict(message)["role"] for message in messages]

        logger.info(
            f"adapter_stage = messages_added session_id: {self.session_id} "
            f"user_id: {self.user_id} added_count: {len(messages)} "
            f"roles: {added_roles} total_messages: {len(current_history)}"
        )

    def clear(self) -> None:
        self.memory_manager.clear()
        logger.info(
            f"adapter_stage = cleared session_id: {self.session_id} user_id: {self.user_id}"
        )
