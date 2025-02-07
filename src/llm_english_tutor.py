import os
from dotenv import load_dotenv
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import trim_messages, get_buffer_string

from langchain_groq import ChatGroq


def connect_api() -> ChatGroq:
    """Obtém a chave da API de variável de ambiente e inicializa o modelo Groq."""
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "A chave da API não foi encontrada. Configure a variável de ambiente GROQ_API_KEY."
        )

    model = "llama-3.3-70b-versatile"
    groq_chat = ChatGroq(groq_api_key=api_key, model_name=model, temperature=0.5)
    return groq_chat


def chatbot_personality():
    """Define a personalidade do chatbot e configura o prompt e a memória."""
    system_prompt = (
        "You are my English tutor. Correct my mistakes and suggest better words."
    )
    history = ChatMessageHistory()

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{human_input}"),
        ]
    )
    return prompt, history


def _count_tokens(messages):
    return len(get_buffer_string(messages).split())


def conversation_with_chatbot(groq_chat, prompt, history, user_question):
    """Inicia uma conversa com o chatbot e retorna a resposta."""
    try:

        trimmed_history = trim_messages(
            history.messages, max_tokens=1024, token_counter=_count_tokens
        )

        final_prompt = prompt.format_messages(
            chat_history=trimmed_history, human_input=user_question
        )

        response = groq_chat.invoke(final_prompt)

        history.add_user_message(user_question)
        history.add_ai_message(response.content)

        return response.content or "Desculpe, não consegui gerar uma resposta."
    except Exception as e:
        return f"Erro: {str(e)}"
