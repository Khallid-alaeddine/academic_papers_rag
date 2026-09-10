import os 
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFDirectoryLoader

llm_model = "gpt-4o-mini"

# Folder path with 4 papers as PDF
script_dir = os.path.dirname(os.path.abspath(__file__))
folder_path = os.path.join(script_dir, "docs")

# Initialize the directory loader 
loader = PyPDFDirectoryLoader(folder_path)

#load all the PDF's
docs = loader.load()

print(f"Loaded {len(docs)} pages")