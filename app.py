import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

st.title("🔥 IC Engine Q&A Assistant")
st.write("Ask any question based on the Internal Combustion Engine technical document.")

@st.cache_resource
def load_knowledge_base():
    loader = PyPDFLoader("topic_info.pdf")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

db = load_knowledge_base()

query = st.text_input("Type your question here (e.g., What is the difference between Otto and Diesel cycles?):")

if query:
    results = db.similarity_search(query, k=2)
    st.subheader("Answer / Relevant Info from Document:")
    for result in results:
        st.info(result.page_content)
