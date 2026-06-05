from config import MODEL_NAME, classifier_prompt, IntentSchema
from handlers import handle_chat, handle_email, handle_summarize, handle_code
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableLambda
from typing import cast

classifier_llm = ChatGroq(model=MODEL_NAME, temperature=0.0)

CLASSIFIER_PROMPT = ChatPromptTemplate.from_messages(
    [("system", classifier_prompt), ("user", "{user_input}")]
)

structured_classifier_llm = classifier_llm.with_structured_output(IntentSchema)

classifier_chain = CLASSIFIER_PROMPT | structured_classifier_llm


def classify_intent(user_input):
    result = cast(IntentSchema, classifier_chain.invoke({"user_input": user_input}))

    return result.intent


handlers = {
    "chat": handle_chat,
    "summarize": handle_summarize,
    "email": handle_email,
    "code": handle_code,
}


intent_branch = RunnableBranch(
    (
        lambda x: x["intent"] == "summarize",  # type: ignore
        RunnableLambda(lambda x: handle_summarize(x["inputs"])),  # type: ignore
    ),
    (
        lambda x: x["intent"] == "email",  # type: ignore
        RunnableLambda(lambda x: handle_email(x["inputs"])),  # type: ignore
    ),
    (
        lambda x: x["intent"] == "code",  # type: ignore
        RunnableLambda(lambda x: handle_code(x["inputs"])),  # type: ignore
    ),
    RunnableLambda(lambda x: handle_chat(x["inputs"])),  # type: ignore
)
