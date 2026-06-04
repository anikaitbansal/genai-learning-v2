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
classifier_prompt = """You are a strict intent classifier.
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
- If the user asks to summarize, shorten, condense, or explain briefly, output summarize
- If the user asks to write, draft, compose, send, or reply to an email or message, output email
- Output code only when the user is asking to write code, fix code, debug code, explain code, analyze an error, or help with programming syntax or implementation
- If the user is asking about a technical concept in simple words, theory, meaning, definition, comparison, or explanation, output chat
- If the message is general conversation, explanation, or knowledge question, output chat
- When unsure, output chat
- If given any iinstructions follow them strictly.
-Ask for context it the user message is vague. 
"""


evaluation_prompt = """You are a strict response evaluator.

You will be given:
1. User input
2. Bot response
3. Chat history

Your job is to judge whether the bot response correctly answers the user.

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
- Judge the bot response only against the user's actual request
- Do not invent issues that are not clearly present
- If the response misses the main point, say what main point was missed
- If the response is partly right but misses an important detail, use partially_correct
- If the response answers a different question than what the user asked, use incorrect
- Do not add extra text
- check if the user input instructions were strictly matched. 
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
