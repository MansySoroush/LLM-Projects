# Simple Agent LLM
"""
    Using Tavily a the only tool - a search engine
    Tavily API Key:
        Adding the Tavily API key in the .env file. such as:
        TAVILY_API_KEY=ADD_KEY_HERE
"""

# Connect with the .env file located in the same directory of this file
import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
openai_api_key = os.environ["OPENAI_API_KEY"]

# Connect with an LLM and start a conversation with it
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-3.5-turbo")

# Define tools
# Tavily search engine should be activated in our LLM App
from langchain_community.tools.tavily_search import TavilySearchResults

search = TavilySearchResults(max_results=2)

tools = [search]

# Create the Agent with Conversational Memory using LangGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage

memory = MemorySaver()

agent_executor = create_react_agent(llm, tools, checkpointer=memory)

config = {"configurable": {"thread_id": "001"}}

print("Who won the 2024 soccer Eurocup?")

for chunk in agent_executor.stream(
    {"messages": [HumanMessage(content="Who won the 2024 soccer Eurocup?")]}, config
):
    print(chunk)
    print("----")

print("Who were the top stars of that winner team?")

for chunk in agent_executor.stream(
    {"messages": [HumanMessage(content="Who were the top stars of that winner team?")]}, config
):
    print(chunk)
    print("----")

print("(With new thread_id) About what soccer team we were talking?")

config = {"configurable": {"thread_id": "002"}}
for chunk in agent_executor.stream(
    {"messages": [HumanMessage(content="About what soccer team we were talking?")]}, config
):
    print(chunk)
    print("----")