from flask import Flask, render_template, jsonify, request
from src.helper import download_hf_embedding
# from langchain.vectorstores import Pinecone
from pinecone import Pinecone
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from src.prompt import *

from langchain_pinecone import PineconeVectorStore

import os

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
index_name = "medical-chatbot-v2"

# print(PINECONE_API_KEY)

embeddings = download_hf_embedding()

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(index_name)

# docsearch = Pinecone.from_existing_index(index_name, embeddings)

docsearch = PineconeVectorStore(
    index=index,
    embedding=embeddings,
    text_key="text",
)

PROMPT=PromptTemplate(template=prompt_template, input_variables=["context", "question"])

chain_type_kwargs = {"prompt": PROMPT}

llm=CTransformers(
  model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
  model_type="llama",
  config={
    'max_new_tokens':512,
    'temperature':0.8
    }
  )

# retriever = docsearch.as_retriever(search_kwargs={"k": 2})
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 2})

qa=RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs
  )

@app.route("/")
def index():
  return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    result=qa({"query": input})
    print("Response : ", result["result"])
    return str(result["result"])


if __name__ == '__main__':
  app.run(host="0.0.0.0", port=8080, debug=True)