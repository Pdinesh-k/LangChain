
import streamlit as st
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langserve import add_routes
import uvicorn
#from langchain_community.llms import Ollama
from fastapi import FastAPI

from dotenv import load_dotenv
load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

app = FastAPI(
    title = "LangChain Server",
    version = "1.0",
    description = "A simple API Server"
)

google_genai_model = ChatGoogleGenerativeAI(model = "models/some_model")

add_routes(
    app,
    google_genai_model,
    path = "/gemini"
)

#llm = Ollama(model = "llama2")

prompt1 = ChatPromptTemplate.from_template("Write me an essay about {topic} in 100 words")
#prompt2 = ChatPromptTemplate.from_template("Write me an poem about {topic} in 100 words")

add_routes(
    app,
    prompt1 | google_genai_model,
    path = "/essay"
)


if __name__ == "__main__":
    uvicorn.run(app,host="localhost",port = 8000)