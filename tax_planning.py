import streamlit as st
import pandas as pd
from langchain.docstore.document import Document
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
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

# Define the enhanced prompt template for tax planning
enhanced_prompt_template = """You are a knowledgeable tax planning assistant. Use the following pieces of context to answer the tax-related question at the end. If you don't know the answer or if the information is not present in the context, just say that you don't have enough information to provide a definitive answer, and suggest consulting a tax professional for personalized advice.

Context:
{context}

Question: {question}

Based on the question, provide the following:
1. Answer to the question
2. Suggestions for tax planning or financial strategies that might be beneficial

Answer and Suggestions:"""

ENHANCED_PROMPT = PromptTemplate(
    template=enhanced_prompt_template,
    input_variables=["context", "question"]
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

# Function to load and process financial knowledge base
def load_financial_knowledge_base(file_path):
    if os.path.isfile(file_path) and file_path.endswith('.pdf'):
        # If it's a single PDF file
        loader = PyPDFLoader(file_path)
        documents = loader.load()
    else:
        raise ValueError(f"Invalid file path: {file_path}. It should be a PDF file.")
    return documents

# Load financial knowledge base
financial_docs = load_financial_knowledge_base("C:\\Download\\unlearn\\Union Budget 2024.pdf")
financial_vector_store = FAISS.from_documents(financial_docs, cohere_embeddings)

# Function to create a VectorStore from documents
def create_vector_store(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(documents)
    # Combine user documents with financial knowledge base
    all_docs = split_docs + financial_docs
    vector = FAISS.from_documents(all_docs, cohere_embeddings)
    return vector

# Function to create a RetrievalQA pipeline
def create_qa_pipeline(vector):
    retriever = vector.as_retriever(search_kwargs={"k": 5})  # Retrieve top 5 most relevant documents
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": ENHANCED_PROMPT}
    )
    return qa_chain

# Streamlit UI
st.set_page_config(page_title="TaxSavvy Stance", page_icon="💼", layout="wide")

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
st.title("💼 TaxSavvy Stance")
st.subheader("Your Intelligent Tax Planning Assistant")

# Initialize conversation history in session state
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

# Function to add to conversation history
def add_to_history(role, content):
    st.session_state.conversation_history.append({"role": role, "content": content})

# Sidebar for document upload
with st.sidebar:
    st.header("📁 Document Upload")
    uploaded_file = st.file_uploader("Upload Tax Documents", type=["pdf", "xlsx", "csv"])
    if uploaded_file:
        st.success("Document uploaded successfully!")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("🤖 TaxSavvy Stance")

    # Display conversation history
    for message in st.session_state.conversation_history:
        if message["role"] == "user":
            st.write("You: " + message["content"])
        else:
            st.write("TaxSavvy Stance: " + message["content"])

    query = st.text_input("Enter your tax-related question:")
    if st.button("Get Answer", key="query_button"):
        if 'vector_store' not in st.session_state or st.session_state.vector_store is None:
            st.warning("Please upload a document first.")
        elif query:
            add_to_history("user", query)
            with st.spinner("Analyzing your question..."):
                qa_pipeline = create_qa_pipeline(st.session_state.vector_store)

                conversation_context = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.conversation_history[-5:]])

                result = qa_pipeline({
                    "conversation_history": conversation_context,
                    "query": query
                })

                add_to_history("assistant", result["result"])
                st.info(result["result"])
        else:
            st.warning("Please enter a question.")

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
st.markdown("*Disclaimer: TaxSavvy Stance provides general tax information. For personalized advice, please consult a qualified tax professional.*")
