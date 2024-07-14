import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.vectorstores import FAISS
import time
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from dotenv import load_dotenv
load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")

if "vector" not in st.session_state:
    st.session_state.web = WebBaseLoader("https://www.thesprucepets.com/persian-cats-gallery-4121944")
    st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    st.session_state.docs = st.session_state.web.load()
    st.session_state.splitter = RecursiveCharacterTextSplitter(chunk_size = 1000,chunk_overlap = 200)
    st.session_state.final_docs = st.session_state.splitter.split_documents(st.session_state.docs)
    st.session_state.vectors = FAISS.from_documents(st.session_state.final_docs , st.session_state.embeddings)

st.title("Gemini - The Boss")
llm = ChatGoogleGenerativeAI(google_api_key=google_api_key,model="gemini-pro")

prompt = ChatPromptTemplate.from_template(

    """
Answer the questions based on the provided question only.
Please provide the most accurate responnse based on the question
<context>
{context}
<context>
Questions:{input}
    """
)
document_chain = create_stuff_documents_chain(llm,prompt)
retreiver = st.session_state.vectors.as_retriever()
retreiver_chain = create_retrieval_chain(retreiver,document_chain)

prompt = st.text_input("Input your prompt here")

if prompt:
    start = time.process_time()
    response = retreiver_chain.invoke({"input":prompt})
    print("Response time :",time.process_time()-start)
    st.write(response["answer"])