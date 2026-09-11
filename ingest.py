import os 
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore

llm_model = "gpt-4o-mini"

# Folder path with 4 papers as PDF
script_dir = os.path.dirname(os.path.abspath(__file__))
folder_path = os.path.join(script_dir, "docs")

# Initialize the directory loader 
loader = PyPDFDirectoryLoader(folder_path)

#load all the PDF's
documents = loader.load()

#print(f"Loaded {len(docs)} pages")

# setting splitter parameters
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
chunks = text_splitter.split_documents(documents)

#print(f"Split into {len(chunks)} chunks")

# embedding text chunks
embeddings = OpenAIEmbeddings()
db = QdrantVectorStore.from_documents(
    chunks,
    embeddings,
    url="http://localhost:6333",
    collection_name="academic_papers"
)