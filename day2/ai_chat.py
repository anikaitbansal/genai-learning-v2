from groq import Groq

client = Groq(api_key="YOUR_GROQ_API_KEY")

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        break

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", 
            "content": "You are a strict technical interviewer. Ask one question at a time. If the user says they don't know the answer, do not ask a new question. Instead, give a small hint and encourage them to try again. Only move to the next question after the user attempts an answer."
            },
            {"role": "user", 
            "content": user_input
            }
        ]
    )

    print("AI:", response.choices[0].message.content)