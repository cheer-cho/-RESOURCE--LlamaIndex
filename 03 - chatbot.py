from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.openai import OpenAI
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.storage.chat_store import SimpleChatStore
import os

load_dotenv()

# Clean up any existing chat history
if os.path.exists("./data/histories/chat_history.json"):
    os.remove("./data/histories/chat_history.json")

# Initialize fresh chat store and memory
chat_store = SimpleChatStore()
memory = ChatMemoryBuffer.from_defaults(
    token_limit=3000,
    chat_store=chat_store,
    chat_store_key="user1"
)

llm = OpenAI(model="gpt-4o-mini")
data = SimpleDirectoryReader("./data/paul_graham/").load_data()
index = VectorStoreIndex.from_documents(data)

chat_engine = index.as_chat_engine(
    chat_mode="best",
    llm=llm,
    memory=memory,
    verbose=True
)

# Save chat history to disk after each interaction
while True:
    text_input = input("User: ")
    if text_input == "exit":
        chat_store.persist("./data/histories/chat_history.json")
        break
    response = chat_engine.chat(text_input)
    print(f"Agent: {response}")