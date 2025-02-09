from fastapi import FastAPI
from langserve import add_routes
import src.llm_english_tutor as llm

model = llm.connect_api()
prompt, history = llm.chatbot_personality()
chain = llm.conversation_with_chatbot(model, prompt, history)

app = FastAPI(title="Meu tutor de inglês", description="Meu tutor de inglês particular desenvolvido com IA.")
add_routes(app, chain, path="/tutor")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
