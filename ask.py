from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
import os
from dotenv import load_dotenv
load_dotenv()

llm_model = "gpt-4o-mini"

# embedding model for upcoming query
embeddings = OpenAIEmbeddings()

#connect to the existing Qdrant collection (read only)
db = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    url="http://localhost:6333",
    collection_name="academic_papers"
)

# query 
qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(temperature=0, model=llm_model),
    chain_type="stuff",
    retriever=db.as_retriever(),
    return_source_documents=True
)
query = "What does AIIE measure? Answer in one sentence."
response = qa.invoke({"query": query})
# Print a response
print(response["result"])

# Loop over the retrieved source chunks and print each one's file name and page number  
print("\nSources:")
for doc in response["source_documents"]:
    filename = os.path.basename(doc.metadata["source"])
    page = doc.metadata.get("page_label", doc.metadata["page"])
    print(f"- {filename}, page {page}")