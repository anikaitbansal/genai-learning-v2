from groq import Groq

client = Groq(api_key="YOUR API KEY HERE") # Replace with your actual API key or set it up in your environment variables for better security.

# messages represent the tone and content of the conversation.
messages = [
    {"role": "system", "content":"You are a strict technical interviewer. Ask follow-up questions and evaluate answers."}
]

while True:

    user_input = input("You (Type (exit) to quit): ")
    messages.append({"role": "user", "content": user_input}) # We add the user's input to the messages list, which maintains the context of the conversation. This allows the AI to generate responses that are relevant to the ongoing dialogue.

    if user_input.lower() == "exit":
        break

    # 2. LIMIT HISTORY HERE 
    messages = [messages[0]] + messages[-9:] #this line limits the conversation history to the last 10 messages. This is important to prevent the context from becoming too large, which can lead to increased latency and higher costs when making API calls. By keeping only the most recent messages, we ensure that the AI has enough context to generate relevant responses while maintaining efficiency.

    #api call
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        temperature=0.3 # Adjust the temperature for more or less randomness or controling the creativity of the response. For diffierent use cases, you might want to experiment with this value.
        # we can also add max_tokens to limit the length of the response, or stop sequences to control when the response should end. 
    )

    bot_reply = response.choices[0].message.content # we do this to extract and store the AI's response from the API's response object. 
    messages.append({"role": "assistant", "content": bot_reply}) # We add the AI's response to the messages list to maintain the conversation history, allowing the AI to generate contextually relevant responses in future interactions.

    print("AI:", bot_reply)