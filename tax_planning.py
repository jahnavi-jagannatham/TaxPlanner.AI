import streamlit as st
import pandas as pd
from langchain.docstore.document import Document
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import CohereEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.llms import Cohere
import cohere
import os
from dotenv import load_dotenv
import tempfile
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

# Set up Cohere API Key
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
cohere_client = cohere.Client(COHERE_API_KEY)

# Initialize Cohere client and embedding model
cohere_embeddings = CohereEmbeddings(model="embed-english-v2.0", cohere_api_key=COHERE_API_KEY)

# Define the prompt template for tax planning
prompt_template = """You are a knowledgeable tax planning assistant. Use the following pieces of context to answer the tax-related question at the end. If you don't know the answer or if the information is not present in the context, just say that you don't have enough information to provide a definitive answer, and suggest consulting a tax professional for personalized advice.

{context}

Question: {question}
Answer:"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

# Initialize the Cohere model for LLM
llm = Cohere(client=cohere_client, model="command-xlarge-nightly")

# Function to load PDF documents
def load_pdf_documents(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_file_path = temp_file.name

    loader = PyPDFLoader(temp_file_path)
    data = loader.load()
    os.unlink(temp_file_path)
    return data

# Function to load Excel documents
def load_excel_documents(uploaded_file):
    if uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
        df = pd.read_excel(uploaded_file)
    elif uploaded_file.type == "text/csv":
        df = pd.read_csv(uploaded_file)
    else:
        st.error("Unsupported file type. Please upload a PDF, Excel, or CSV file.")
        st.stop()

    documents = []
    for index, row in df.iterrows():
        text = " ".join(str(value) for value in row.values)
        document = Document(page_content=text)
        documents.append(document)
    return documents

# Function to create a VectorStore from documents
def create_vector_store(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(documents)
    vector = FAISS.from_documents(split_docs, cohere_embeddings)
    return vector

# Function to create a RetrievalQA pipeline
def create_qa_pipeline(vector):
    retriever = vector.as_retriever(search_kwargs={"k": 5})  # Retrieve top 5 most relevant documents
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT}
    )
    return qa_chain

# Streamlit UI
st.set_page_config(page_title="TaxPlannerAI", page_icon="💼", layout="wide")

# Custom CSS to improve aesthetics
st.markdown("""
    <style>
    .main {
        background-color: #000000;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .st-bw {
        background-color: #000000;
    }
    </style>
    """, unsafe_allow_html=True)

# Header
st.title("💼 TaxPlannerAI")
st.subheader("Your Intelligent Tax Planning Assistant")

# Sidebar for document upload
with st.sidebar:
    st.header("📁 Document Upload")
    uploaded_file = st.file_uploader("Upload Tax Documents", type=["pdf", "xlsx", "csv"])
    if uploaded_file:
        st.success("Document uploaded successfully!")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("🤖 Ask TaxPlannerAI")
    query = st.text_input("Enter your tax-related question:")
    if st.button("Get Answer", key="query_button"):
        if 'vector_store' not in st.session_state or st.session_state.vector_store is None:
            st.warning("Please upload a document first.")
        elif query:
            with st.spinner("Analyzing your question..."):
                qa_pipeline = create_qa_pipeline(st.session_state.vector_store)
                result = qa_pipeline({"query": query})
                st.info(result["result"])
        else:
            st.warning("Please enter a question.")

with col2:
    st.header("💡 Tax Planning Tips")
    st.markdown("""
    - Keep track of all your income sources
    - Understand deductible expenses
    - Maximize your retirement contributions
    - Consider tax-efficient investments
    - Stay updated on tax law changes
    """)

# Document processing
if uploaded_file is not None and 'vector_store' not in st.session_state:
    with st.spinner("Processing your document..."):
        if uploaded_file.type == "application/pdf":
            documents = load_pdf_documents(uploaded_file)
        else:
            documents = load_excel_documents(uploaded_file)
        st.session_state.vector_store = create_vector_store(documents)
        st.success("Document processed and ready for questions!")

# Footer
st.markdown("---")
st.markdown("*Disclaimer: TaxPlannerAI provides general tax information. For personalized advice, please consult a qualified tax professional.*")