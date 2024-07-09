# Load model directly
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/blenderbot-400M-distill")

# Function to start the chat
def chat():
    conversation_history = []

    while True:
        user_input = input("You: ")

        if user_input.lower() in ["exit", "quit"]:
            print("Chatbot: Goodbye!")
            break

        # Append user input to conversation history
        conversation_history.append(user_input)

           # Generate model response
        inputs = tokenizer.encode(" ".join(conversation_history[-1:]) + tokenizer.eos_token, return_tensors="pt")
        reply_ids = model.generate(inputs, max_length=100, num_return_sequences=1, temperature=0.9, top_p=0.9, repetition_penalty=1.2)
        chatbot_response = tokenizer.decode(reply_ids[:, inputs.shape[0]:][0], skip_special_tokens=True)
        # Print model response
        print("Chatbot:", chatbot_response)

# Start the chat
chat() 