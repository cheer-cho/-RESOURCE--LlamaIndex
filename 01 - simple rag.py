from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.openai import OpenAI

load_dotenv()

# Initialize OpenAI with GPT-4
OpenAI(model="gpt-4o-mini")

# Load documents from the data directory
documents = SimpleDirectoryReader("data").load_data()

# Create a vector store index from the documents
index = VectorStoreIndex.from_documents(documents)

# Create a query engine
query_engine = index.as_query_engine()

# Query the documents
response = query_engine.query("What are the design goals and please give details about them!")

print(response)