from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain.embeddings import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_community.vectorstores import Pinecone as Pine
import os

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY") # or set via environment outside script
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_API_ENV = "gcp-starter"
index_name = "medical-chatbot"

# 2. Load PDF documents
pdf_dir = "./data"
loader = PyPDFDirectoryLoader(pdf_dir)
documents = loader.load()

# 3. Split documents into text chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # Experiment as needed
    chunk_overlap=20
)
text_chunks = splitter.split_documents(documents)

texts = [chunk.page_content for chunk in text_chunks]

def download_hf_embedding():
  embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
  )
  return embeddings

embeddings = download_hf_embedding()

pc = Pinecone(api_key=PINECONE_API_KEY)

if index_name not in [index.name for index in pc.list_indexes()]:
    # Create a serverless index; metric and dimension must be provided
    pc.create_index(
        index_name,
        dimension=384,  # Must match your vector dimension!
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
# Connect to index
index = pc.Index(index_name)

# 6. Prepare data and upsert to Pinecone
    # Step 1: Generate embeddings for all your texts
embedding_vectors = embeddings.embed_documents(texts)

vectors = [
    (f"doc-{i}", embedding_vectors[i], {"text": texts[i]})
    for i in range(len(texts))
]

batch_size = 100  # start small, can increase as long as under 4MB
for i in range(0, len(vectors), batch_size):
    batch = vectors[i:i+batch_size]
    index.upsert(vectors=batch)
    
print("✅ PDF processing and upsert complete.")

docsearch = Pine(
    index,          # your pinecone.Index object
    embeddings,          # your embedding function instance
    texts               # the name of the metadata field used for text
)

query = "What are allergies?"  # must be str
results = docsearch.similarity_search(query, k=3)