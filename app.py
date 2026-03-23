import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub
from dotenv import load_dotenv
import os

load_dotenv()


print("hello")
print("hello")
st.title("📄 PDF Q&A Chatbot")

# Upload PDF
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file:
    # Save file
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    # Load PDF
    loader = PyPDFLoader("temp.pdf")
    documents = loader.load()

    # Split text
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = splitter.split_documents(documents)

    # Embeddings
    embeddings = HuggingFaceEmbeddings()

    # Vector DB
    db = FAISS.from_documents(texts, embeddings)

    # LLM
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-large",
        model_kwargs={"temperature": 0.5, "max_length": 512}
    )

    # QA Chain
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=db.as_retriever()
    )

    st.success("PDF loaded! Ask your question 👇")

    # User question
    query = st.text_input("Ask a question:")

    if query:
        answer = qa.run(query)
        st.write("🤖 Answer:", answer) 