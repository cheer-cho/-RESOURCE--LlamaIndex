import os
import nest_asyncio
from dotenv import load_dotenv

load_dotenv()

nest_asyncio.apply()

from llama_parse import LlamaParse
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

LLAMA_INDEX_CLOUD_KEY = os.getenv("LLAMA_INDEX_CLOUD_KEY")

parser = LlamaParse(
  api_key=LLAMA_INDEX_CLOUD_KEY,
  result_type="markdown",
  verbose=True,
)

file_extractor = {".pdf": parser}

documents = SimpleDirectoryReader("./pdf", file_extractor=file_extractor).load_data()

index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine()

response = query_engine.query("What is the main idea of the document?")

print(response)
