from git import Repo
from dotenv import load_dotenv
import os
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import gradio as gr

# Load API key dari file .env
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# URL repo GitHub kamu
REPO_URL = "https://github.com/novdev72/chatbot"

# Clone repo ke folder lokal 'data'
if not os.path.exists("data"):
    Repo.clone_from(REPO_URL, "data/")

# Load file README.md
loader = TextLoader("data/README.md")
docs = loader.load()

# Potong dokumen jadi chunk
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_documents(docs)

# Embedding dan vectorstore
embedding = OpenAIEmbeddings(openai_api_key=api_key)
vectorstore = FAISS.from_documents(chunks, embedding)

# Buat chatbot retrieval
qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(openai_api_key=api_key),
    retriever=vectorstore.as_retriever()
)

# UI Gradio
def chat_fn(message):
    return qa.run(message)

gr.ChatInterface(fn=chat_fn).launch()
