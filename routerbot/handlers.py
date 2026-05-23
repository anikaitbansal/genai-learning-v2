from uuid import UUID
from config import MODEL_NAME
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import ConfigurableFieldSpec
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_memory_adapter import LangChainMemoryAdapter

handler_llm = ChatGroq(model=MODEL_NAME, temperature=0.2)

SYSTEM_PROMPTS = {
    "chat": "You are a helpful assistant with access to a knowledge base that may include user-uploaded documents. When a user asks about a document or uploaded file, use the retrieved context — it contains content extracted from that document. Use previous conversation context when relevant.",
    "summarize": "You summarize text clearly and concisely. If the user asks to summarize an uploaded document, use the retrieved context which contains the document's content. Use previous context if the user is referring to earlier text.",
    "email": "You are a professional email writer. If the user is refining a previously written email, use conversation context.",
    "code": "You help debug and explain code clearly. Use previous conversation context if the user is referring to earlier code.",
}

output_parser = StrOutputParser()


def build_rag_prompt(base_prompt, retrieved_chunks):
    if not retrieved_chunks:
        return base_prompt

    context_lines = []
    for index, chunk in enumerate(retrieved_chunks, start=1):
        context_lines.append(
            f"Context {index}:\n"
            f"Title: {chunk['title']}\n"
            f"Content: {chunk['content']}\n"
        )

    joined_context = "\n\n".join(context_lines)

    return (
        f"{base_prompt}\n\n"
        f"The following context was retrieved from the knowledge base, which may include user-uploaded documents (PDFs and files).\n"
        f"Use this context to answer the user's question — if the user is asking about a document or uploaded file, this context contains its content.\n"
        f"If the context helps, use it to answer more accurately.\n"
        f"If the context is genuinely unrelated to the question, you may answer from general knowledge instead.\n"
        f"Synthesize the information in your own words; do not copy verbatim unless the user asks for an exact quote.\n\n"
        f"Retrieved Context:\n{joined_context}"
    )


def get_message_history(session_id: str, user_id: UUID) -> BaseChatMessageHistory:
    return LangChainMemoryAdapter(session_id=session_id, user_id=user_id)


HISTORY_FACTORY_CONFIG = [
    ConfigurableFieldSpec(
        id="session_id",
        annotation=str,
        name="Session ID",
        description="Unique identifier for the chat session.",
        default="",
        is_shared=True,
    ),
    ConfigurableFieldSpec(
        id="user_id",
        annotation=UUID,
        name="User ID",
        description="Authenticated user owning the session.",
        default=None,
        is_shared=True,
    ),
]


def build_chain_with_history(system_prompt_text: str):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "{system_prompt}"),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{user_input}"),
        ]
    )

    base_chain = prompt | handler_llm | output_parser

    chain_with_history = RunnableWithMessageHistory(
        base_chain,
        get_message_history,
        input_messages_key="user_input",
        history_messages_key="history",
        history_factory_config=HISTORY_FACTORY_CONFIG,
    )

    return chain_with_history


CHAINS_BY_INTENT = {
    intent: build_chain_with_history(system_prompt_text)
    for intent, system_prompt_text in SYSTEM_PROMPTS.items()
}


def generate_response(
    intent: str,
    user_input: str,
    session_id: str,
    user_id: UUID,
    retrieved_chunks=None,
    retry_reason: str = "",
    retry_count: int = 0,
):
    base_system_prompt = SYSTEM_PROMPTS[intent]
    final_system_prompt = build_rag_prompt(base_system_prompt, retrieved_chunks or [])

    if retry_count == 1 and retry_reason.strip():
        final_system_prompt = (
            f"{final_system_prompt}\n\n"
            f"The previous answer was not good enough.\n"
            f"Issue to fix: {retry_reason.strip()}\n"
            f"Answer the user's original request again more accurately.\n"
            f"Strictly fix the issue above."
        )

    chain = CHAINS_BY_INTENT[intent]

    bot_response = chain.invoke(
        {
            "system_prompt": final_system_prompt,
            "user_input": user_input,
        },
        config={
            "configurable": {
                "session_id": session_id,
                "user_id": user_id,
            }
        },
    )

    return bot_response


def handle_chat(
    user_input,
    session_id,
    user_id,
    retrieved_chunks=None,
    retry_reason="",
    retry_count=0,
):
    return generate_response(
        "chat",
        user_input,
        session_id,
        user_id,
        retrieved_chunks,
        retry_reason,
        retry_count,
    )


def handle_email(
    user_input,
    session_id,
    user_id,
    retrieved_chunks=None,
    retry_reason="",
    retry_count=0,
):
    return generate_response(
        "email",
        user_input,
        session_id,
        user_id,
        retrieved_chunks,
        retry_reason,
        retry_count,
    )


def handle_summarize(
    user_input,
    session_id,
    user_id,
    retrieved_chunks=None,
    retry_reason="",
    retry_count=0,
):
    return generate_response(
        "summarize",
        user_input,
        session_id,
        user_id,
        retrieved_chunks,
        retry_reason,
        retry_count,
    )


def handle_code(
    user_input,
    session_id,
    user_id,
    retrieved_chunks=None,
    retry_reason="",
    retry_count=0,
):
    return generate_response(
        "code",
        user_input,
        session_id,
        user_id,
        retrieved_chunks,
        retry_reason,
        retry_count,
    )
