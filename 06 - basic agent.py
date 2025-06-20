import os
from dotenv import load_dotenv
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
from llama_index.core.tools import FunctionTool

load_dotenv()

llm = OpenAI(model="gpt-4o")

def add (a: float, b: float) -> float:
    """Add two numbers and return the sum"""
    return a + b

def subtract (a: float, b: float) -> float:
    """Subtract two numbers and return the difference"""
    return a - b

def multiply (a: float, b: float) -> float:
    """Multiply two numbers and return the product"""
    return a * b

def divide (a: float, b: float) -> float:
    """Divide two numbers and return the quotient"""
    return a / b

add_tool = FunctionTool.from_defaults(fn=add)
subtract_tool = FunctionTool.from_defaults(fn=subtract)
multiply_tool = FunctionTool.from_defaults(fn=multiply)
divide_tool = FunctionTool.from_defaults(fn=divide)

agent = ReActAgent.from_tools([add_tool, subtract_tool, multiply_tool, divide_tool], llm=llm, verbose=True)

response = agent.chat("What is 10 + 5?")
print(response)

response = agent.chat("What is 10 + 5?  Use the tools to solve the problem.")
print(response)

response = agent.chat("What is 20 + (2 / 4) * 5 - 10? Use the tools to solve the problem.")
print(response)
