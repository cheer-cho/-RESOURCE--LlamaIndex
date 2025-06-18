from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.openai import OpenAI

load_dotenv()

# Initialize OpenAI with GPT-4o-mini
llm = OpenAI(model="gpt-4o-mini")

# Load data from the data directory
data = SimpleDirectoryReader("./data/paul_graham/").load_data()
index = VectorStoreIndex.from_documents(data)

chat_engine = index.as_chat_engine(chat_mode="best", llm=llm, verbose=True)

response = chat_engine.chat("What are the first programs Paul Graham tried writing?")

print(response)