from groq import Groq
import os
from typing import Literal
from pydantic import BaseModel, Field

client = (
    Groq()
)  # This creates the connection between your Python code and the Groq API. i don not need to pass the api key here as i have set it up in my environment variables. this is so when i push this code to github, i do not accidentally expose my api key. env is created by running this command in terminal: setx GROQ_API_KEY "gsk_abc123xyz"


QDRANT_URL = os.environ.get("QDRANT_URL", "")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "")
QDRANT_COLLECTION_NAME = "chatbot_chunks"

COHERE_API_KEY = os.environ.get("COHERE_API_KEY", "")
COHERE_RERANK_MODEL = "rerank-v3.5"

MODEL_NAME = "llama-3.1-8b-instant"
EVALUATOR_MODEL_NAME = "llama-3.1-8b-instant"

RAG_KNOWLEDGE_FILE = "knowledge_base.json"
RAG_TOP_K = 7
RAG_CHUNK_SIZE = 800
RAG_CHUNK_OVERLAP = 120
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
RAG_SIMILARITY_THRESHOLD = 0.30

# This code sets up a simple chatbot that classifies user input into different intents (chat, summarize, email, code) and handles each intent with a specific function. The conversation history is maintained to provide context for the AI's responses. The user can exit the program by typing "exit".
classifier_prompt = """
You are a strict intent classifier.

Classify the user's message into exactly one of these labels only:
chat
summarize
email
code

Rules:
- Allowed outputs are only: chat, summarize, email, code
- Return only one label
- Do not explain
- Do not add punctuation
- Do not add extra words
- Do not add new lines

Classification rules:
- Classify by the ACTION the user wants, not the topic it is about
- If the user asks to summarize, shorten, or condense something, output summarize (even if the content is code or an email)
- If the user asks to write, draft, compose, send, or reply to an email or message, output email
- Output code only when the user wants to write, fix, debug, explain, or analyze code, OR resolve a programming error or syntax issue
- If the user asks about a technical concept in simple words, theory, meaning, definition, comparison, or explanation, output chat
- If the message is general conversation, explanation, or a knowledge question, output chat
- When unsure, output chat
"""


evaluation_prompt = """
You are a fair but careful response evaluator.

You will be given:

1. User input

2. Bot response

3. Chat history

Your job is to judge whether the bot response reasonably answers the user's request, using chat history when it is relevant.

Return output in exactly this format only:

score: correct

reason: <short reason>

OR

score: partially_correct

reason: <short reason>

OR

score: incorrect

reason: <short reason>

Rules:

- Allowed scores are only: correct, partially_correct, incorrect

- Keep the reason short, specific, and practical

- Judge the bot response against the user's actual request

- Use the provided chat history if the answer depends on earlier conversation

- Do not mark a memory-based answer incorrect just because the current message alone lacks context

Scoring guidance:

- Give the response the benefit of the doubt: a correct answer in an unexpected style, format, or wording is still correct

- A different but valid way of answering is NOT a reason to lower the score

- Use correct when the response answers the user's request, even if it could be phrased better

- Use partially_correct only when a genuinely important part of the request is missing or wrong

- Use incorrect only when the response is factually wrong, off-topic, or answers a different question than what was asked

- Do not penalize extra helpful detail unless the user explicitly asked for brevity and it was ignored

- When in doubt between two scores, choose the higher one

- Do not add extra text
"""


MULTI_QUERY_PROMPT_TEMPLATE = (
    "You are an AI assistant. Your task is to generate exactly 3 different "
    "rephrasings of the user's question to help retrieve relevant documents "
    "from a vector database.\n\n"
    "Strict output rules:\n"
    "- Output EXACTLY 3 lines\n"
    "- One rephrased question per line\n"
    "- No numbering, no bullets, no preamble, no explanations\n"
    "- No blank lines\n"
    "- Each line must be a complete standalone question\n\n"
    "Original question: {question}\n\n"
    "Three rephrased questions:"
)


class IntentSchema(BaseModel):
    intent: Literal["chat", "summarize", "email", "code"] = Field(
        description="The single intent label that best matches the user's message."
    )


class EvaluationSchema(BaseModel):
    score: Literal["correct", "partially_correct", "incorrect"] = Field(
        description="The judgment of how well the bot response answers the user's request."
    )
    reason: str = Field(description="A short, specific reason explaining the score.")
