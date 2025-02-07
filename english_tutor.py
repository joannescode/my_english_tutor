from src.llm_english_tutor import (
    connect_api,
    chatbot_personality,
    conversation_with_chatbot,
)

if __name__ == "__main__":
    groq_chat = connect_api()
    prompt, history = chatbot_personality()

    while True:
        user_question = input("Você: ")
        if user_question.lower() in ["sair", "exit"]:
            break

        response = conversation_with_chatbot(groq_chat, prompt, history, user_question)
        print(f"Tutor: {response}", end="\n")
