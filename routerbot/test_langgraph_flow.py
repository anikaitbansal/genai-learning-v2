from memory_manager import MemoryManager
from retriever import FAISSRetriever
from langgraph_flow import build_langgraph_flow


def main():
    session_id = "test-session"

    memory = MemoryManager(session_id)
    retriever = FAISSRetriever()
    graph = build_langgraph_flow()

    initial_state = {
        "original_message": "who is anikait bansal and where does he live?",
        "chat_history": memory.load(),
        "use_rag": True,
        "retriever": retriever,
        "intent": "",
        "retrieved_chunks": [],
        "rag_used": False,
        "bot_reply": "",
        "evaluation": {},
        "evaluation_reason": "",
        "retry_count": 0
    }

    final_state = graph.invoke(initial_state)

    print("\nLANGGRAPH DEBUG SUMMARY\n")
    print(f"intent: {final_state['intent']}")
    print(f"rag_used: {final_state['rag_used']}")
    print(f"retry_count: {final_state['retry_count']}")
    print(f"retry_happened: {final_state['retry_count'] > 0}")
    print(f"evaluation_score: {final_state['evaluation'].get('score')}")
    print(f"evaluation_reason: {final_state['evaluation'].get('reason')}")
    print(f"retrieved_chunks_count: {len(final_state['retrieved_chunks'])}")

    print("\nFINAL BOT REPLY\n")
    print(final_state["bot_reply"])

    print("\nFULL RESULT\n")
    result = {
        "user_message": initial_state["original_message"],
        "bot_reply": final_state["bot_reply"],
        "intent": final_state["intent"],
        "session_id": session_id,
        "rag_used": final_state["rag_used"],
        "retrieved_chunks": final_state["retrieved_chunks"],
        "evaluation": final_state["evaluation"],
        "retry_count": final_state["retry_count"],
        "retry_happened": final_state["retry_count"] > 0
    }

    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()