import os
from dotenv import load_dotenv
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor

from phoenix.otel import register

load_dotenv()

PHOENIX_API_KEY = os.getenv("PHOENIX_API_KEY")
os.environ["PHOENIX_CLIENT_HEADERS"] = f"api_key={PHOENIX_API_KEY}"
os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = "https://app.phoenix.arize.com"

tracer_provider = register(project_name="LLamaIndex_Observe")
LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)

# RAG Pipeline

# 1. Load data
documents = SimpleDirectoryReader("./data/paul_graham/").load_data()

# 2. Load index from storage if it exists, otherwise create a new index
if os.path.exists("./data/storage"):
    print("Loading index from storage...")
    storage_context = StorageContext.from_defaults(persist_dir="./data/storage")
    index = load_index_from_storage(storage_context)
else:
    print("Creating index from scratch...")
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist("./data/storage")

query_engine = index.as_query_engine()

response = query_engine.query(
    "What is Authogen Studio? Please give as many details as possible."
)
print(response)
