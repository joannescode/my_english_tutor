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
from langchain_core.output_parsers import StrOutputParser

from langchain_groq import ChatGroq


def _local_api_key():
    """Obtém a chave da API através da .env."""
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "\nA chave da API não foi encontrada. Configure a variável de ambiente GROQ_API_KEY."
        )
    return api_key


def _input_api_key():
    """Obtém a chave da API via terminal inserido manualmente."""
    print("\nPor favor passe sua chave de API da 'https://console.groq.com/keys'\n")
    api_key = input(str())
    return api_key


def connect_api() -> ChatGroq:
    """Obtém a chave da API de variável de ambiente ou via terminal e inicializa o modelo Groq."""
    message = """Por favor, escolha o método para adicionar sua chave da API da Groq para conexão com o modelo Llama,
sendo 1 para API local (utilizando .env) ou 2 para adicionar diretamente via terminal."""

    while True:
        print(message, end="")
        user_format_key = input(str())
        if user_format_key == "1":
            api_key = _local_api_key()
            break
        elif user_format_key == "2":
            api_key = _input_api_key()
            break

    model = ChatGroq(groq_api_key=api_key, model="llama-3.3-70b-versatile", temperature=0.5)
    return model


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


def conversation_with_chatbot(model, prompt, history, user_question=None):
    """Inicia uma conversa com o chatbot e retorna a resposta."""
    try:
        # Ajusta o histórico para não ultrapassar o limite de tokens
        trimmed_history = trim_messages(
            history.messages, max_tokens=1024, token_counter=_count_tokens
        )
        parser = StrOutputParser()
        chain = prompt | model | parser

        if user_question is not None:
            response = chain.invoke({"chat_history": trimmed_history, "human_input": user_question})

            history.add_user_message(user_question)
            history.add_ai_message(response)
            return response
        else:
            return chain

    except Exception as e:
        return f"Erro: {str(e)}"
