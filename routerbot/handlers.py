from config import MODEL_NAME
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

handler_llm = ChatGroq(model=MODEL_NAME, temperature=0.2)

SYSTEM_PROMPTS = {
    "chat": "You are a helpful assistant. Answer the user's question directly using your general knowledge. If retrieved context from the knowledge base is provided, use it — it may contain content from user-uploaded documents. If no context is provided, just answer normally from what you know; do not say information is missing or ask for an uploaded document. Use previous conversation context when relevant.",
    "summarize": "You summarize text clearly and concisely. Summarize whatever the user provides: their message text, earlier conversation, or retrieved document context if it is present. Do not ask for a document or say one is missing — just summarize the material available to you. Only use retrieved context if it is actually provided.",
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
        f"If the user is asking about a document or uploaded file, this context contains its content — use it to answer.\n"
        f"If the context is relevant, use it to answer more accurately.\n"
        f"If the context is genuinely unrelated to the question, ignore it and answer from general knowledge — do not force the context into your answer or mention that it was unrelated.\n"
        f"Synthesize the information in your own words; do not copy verbatim unless the user asks for an exact quote.\n\n"
        f"Retrieved Context:\n{joined_context}"
    )


def build_plain_chain(system_prompt_text: str):
    prompt = ChatPromptTemplate.from_messages(
        [("system", "{system_prompt}"), ("human", "{user_input}")]
    )

    chain = prompt | handler_llm | output_parser

    return chain


CHAINS_BY_INTENT = {
    intent: build_plain_chain(system_prompt_text)
    for intent, system_prompt_text in SYSTEM_PROMPTS.items()
}


def generate_response(intent: str, inputs: dict):
    user_input = inputs["user_input"]
    retrieved_chunks = inputs.get("retrieved_chunks")
    retry_reason = inputs.get("retry_reason", "")
    retry_count = inputs.get("retry_count", 0)

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
        }
    )

    return bot_response


def handle_chat(inputs: dict):
    return generate_response("chat", inputs)


def handle_email(inputs: dict):
    return generate_response("email", inputs)


def handle_summarize(inputs: dict):
    return generate_response("summarize", inputs)


def handle_code(inputs: dict):
    return generate_response("code", inputs)
