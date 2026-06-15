import logging
from config import EVALUATOR_MODEL_NAME, evaluation_prompt, EvaluationSchema
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from typing import cast

logger = logging.getLogger(__name__)

evaluator_llm = ChatGroq(model=EVALUATOR_MODEL_NAME, temperature=0.0)

EVALUATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", evaluation_prompt),
        (
            "user",
            "User Input:\n{user_input}\n\nChat History:\n{chat_history}\n\nBot Response:\n{bot_response}",
        ),
    ]
)

structured_evaluator_llm = evaluator_llm.with_structured_output(EvaluationSchema)

evaluation_chain = EVALUATION_PROMPT | structured_evaluator_llm


def build_history_text(chat_history):
    if not chat_history:
        return ""

    history_lines = []
    for message in chat_history:
        history_lines.append(f"{message['role']}: {message['content']}")

    return "\n".join(history_lines)


class ResponseEvaluator:
    def evaluate(self, user_input, bot_response, chat_history=None):
        history_text = build_history_text(chat_history or [])

        try:
            result = cast(
                EvaluationSchema,
                evaluation_chain.invoke(
                    {
                        "user_input": user_input,
                        "chat_history": history_text,
                        "bot_response": bot_response,
                    }
                ),
            )

            logger.info(
                f"evaluation_stage = done score: {result.score} reason: {result.reason}"
            )

            return {"score": result.score, "reason": result.reason}

        except Exception as error:
            logger.exception(f"evaluation_stage = failed error: {str(error)}")
            return {"score": "incorrect", "reason": "Evaluation request failed."}
