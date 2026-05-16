from config import MODEL_NAME, classifier_prompt
from handlers import handle_chat, handle_summarize, handle_email, handle_code
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

classifier_llm = ChatGroq(
    model=MODEL_NAME,
    temperature=0.0
)

CLASSIFIER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", classifier_prompt),
    ("user", "{input}")
])

classifier_parser = StrOutputParser()

classifier_chain = CLASSIFIER_PROMPT | classifier_llm | classifier_parser


def classify_intent(user_input):
    raw_intent = classifier_chain.invoke({
        "input": user_input
    }).strip().lower()

    intent = raw_intent.split()[0]

    valid_intents = ["chat", "summarize", "email", "code"]

    if intent not in valid_intents:
        intent = "chat"

    return intent


handlers = {
    "chat": handle_chat,
    "summarize": handle_summarize,
    "email": handle_email,
    "code": handle_code
}