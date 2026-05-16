import logging
from config import EVALUATOR_MODEL_NAME, evaluation_prompt
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger(__name__)

evaluator_llm = ChatGroq(
    model=EVALUATOR_MODEL_NAME,
    temperature=0.0
)

EVALUATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", evaluation_prompt),
    ("user", "User Input:\n{user_input}\n\nChat History:\n{chat_history}\n\nBot Response:\n{bot_response}")
])

evaluation_parser = StrOutputParser()

evaluation_chain = EVALUATION_PROMPT | evaluator_llm | evaluation_parser

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
            raw_output = evaluation_chain.invoke({
                "user_input": user_input,
                "chat_history": history_text,
                "bot_response": bot_response
            }).strip()

            logger.info("evaluation_stage=raw_output raw_output=%s", raw_output)

            score = "incorrect"
            reason = "Evaluation parsing failed."

            for line in raw_output.splitlines():
                cleaned_line = line.strip()

                if cleaned_line.lower().startswith("score:"):
                    score = cleaned_line.split(":", 1)[1].strip().lower()

                elif cleaned_line.lower().startswith("reason:"):
                    reason = cleaned_line.split(":", 1)[1].strip()

            valid_scores = {"correct", "partially_correct", "incorrect"}

            if score not in valid_scores:
                score = "incorrect"
                reason = "Evaluator returned an invalid score."

            return {
                "score": score,
                "reason": reason
            }
        
        except Exception as error:
            logger.exception("evaluation_stage=failed error=%s", str(error))
            return {
                "score": "incorrect",
                "reason": "Evaluation request failed."
            }