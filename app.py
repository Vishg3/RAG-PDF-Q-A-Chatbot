import streamlit as st
import os
from io import BytesIO
import tempfile
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from dotenv import load_dotenv
load_dotenv()

groq_api_key=os.getenv("GROQ_API_KEY")
os.environ["HF_TOKEN"]=os.getenv("HF_TOKEN")
llm=ChatGroq(groq_api_key=groq_api_key,model_name="Llama3-8b-8192")

prompt=ChatPromptTemplate.from_template(
    """
    Answer the question based on the provided context only.
    Provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Question:{input}
"""
)

def create_vector_embedding(temp_file_path):
    if "vectors" not in st.session_state:
        st.session_state.embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2") 
        st.session_state.loader=PyPDFLoader(temp_file_path)
        st.session_state.docs=st.session_state.loader.load()
        st.session_state.splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        st.session_state.final_docs=st.session_state.splitter.split_documents(st.session_state.docs)
        st.session_state.vectors=FAISS.from_documents(st.session_state.final_docs,st.session_state.embeddings)

st.title("RAG PDF Document Q&A")

uploaded_file=st.file_uploader("Choose a PDF file", type="pdf")
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_file_path=tmp_file.name
user_prompt=st.text_input("Ask me anything about the PDF document")

if uploaded_file and st.button("Document embedding"):
    create_vector_embedding(temp_file_path)
    st.write("Vector database is ready")

import time
if user_prompt:
    document_chain=create_stuff_documents_chain(llm,prompt)
    retriever=st.session_state.vectors.as_retriever()
    retrieval_chain=create_retrieval_chain(retriever,document_chain)

    start=time.process_time()
    response=retrieval_chain.invoke({"input":user_prompt})
    end=time.process_time()
    response_time=end-start
    st.write(f"Response time: {response_time}")
    st.write(response["answer"])

    with st.expander("Document similarity search"):
        for i,doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("-----------------")



    

