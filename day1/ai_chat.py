from groq import Groq

client = Groq(api_key="YOUR_GROQ_API_KEY")

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        break

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "user", "content": user_input}
        ]
    )

    print("AI:", response.choices[0].message.content)