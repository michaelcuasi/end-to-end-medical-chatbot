from src.helper import load_pdfs_from_directory, split_documents_into_chunks, download_hf_embedding
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# PINECONE_API_ENV = "gcp-starter"
index_name = "medical-chatbot-v2"

documents = load_pdfs_from_directory("./data")

text_chunks = split_documents_into_chunks(documents)

texts = [chunk.page_content for chunk in text_chunks]

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
