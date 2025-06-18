# Rag Pipeline

# 1. Load data
# 2. Create index
# 3. Create vector store index with Chroma
# 4. Create query engine

import os
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader

load_dotenv()

# 1. Load data
documents = SimpleDirectoryReader("./data/paul_graham/").load_data()

# 2. Create index
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex, StorageContext

db = chromadb.PersistentClient(path="./data/chroma_db")

chroma_collection = db.get_or_create_collection("quickstart")

vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# 3. Create vector store index with Chroma
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
    show_progress=True,
)

# 4. Create query engine
from llama_index.core import get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor

retriever = VectorIndexRetriever(index=index, similarity_top_k=10)

response_synthesizer = get_response_synthesizer()

query_engine = RetrieverQueryEngine(
  retriever=retriever,
  response_synthesizer=response_synthesizer,
  node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.5)]
)

response = query_engine.query("What is the meaning of life?")
print("**********")
print(response)