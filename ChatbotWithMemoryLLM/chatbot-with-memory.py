# Trick to avoid the nasty deprecation warnings from LangChain
import warnings
from langchain._api import LangChainDeprecationWarning

warnings.simplefilter("ignore", category=LangChainDeprecationWarning)

# Connect with the .env file located in the same directory of this file
import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
openai_api_key = os.environ["OPENAI_API_KEY"]

# Connect with an LLM and start a conversation with it
from langchain_openai import ChatOpenAI

chatbot = ChatOpenAI(model="gpt-3.5-turbo")

from langchain_core.messages import HumanMessage

messagesToTheChatbot = [
    HumanMessage(content="My favorite color is blue."),
]

# Call the ChatModel (the LLM)
response = chatbot.invoke(messagesToTheChatbot)

print("\n----------\n")
print("My favorite color is blue.")
print("\n----------\n")
print(response.content)
print("\n----------\n")

# Check if the Chatbot remembers your favorite color.
response = chatbot.invoke([
    HumanMessage(content="What is my favorite color?")
])

print("\n----------\n")
print("What is my favorite color?")
print("\n----------\n")
print(response.content)
print("\n----------\n")

# Let's add memory to our Chatbot
"""
    Using the ChatMessageHistory package.
    Saving the Chatbot memory in a python dictionary called chatbotMemory.
    Defining the get_session_history function to create a session_id for each conversation.
    Using the built-in runnable RunnableWithMesageHistory.
"""

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# 1. To Create a dictionary to store our memory
chatbotMemory = {}

# 2. To Declare a function to restore and save the message by session id as the key (Separate memory for separate users)
# input: session_id, output: chatbotMemory[session_id]
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in chatbotMemory:
        chatbotMemory[session_id] = ChatMessageHistory()
    return chatbotMemory[session_id]

# 3. To Create a chatbot with message history using RunnableWithMessageHistory
chatbot_with_message_history = RunnableWithMessageHistory(
    chatbot, 
    get_session_history
)

# To create a chat memory for one user session, let's call it session1:
session1 = {"configurable": {"session_id": "001"}}

responseFromChatbot = chatbot_with_message_history.invoke(
    [HumanMessage(content="My favorite color is red.")],
    config=session1,
)

print("\n----------\n")
print("My favorite color is red.")
print("\n----------\n")
print(responseFromChatbot.content)
print("\n----------\n")

responseFromChatbot = chatbot_with_message_history.invoke(
    [HumanMessage(content="What's my favorite color?")],
    config=session1,
)

print("\n----------\n")
print("What's my favorite color? (in session1)")
print("\n----------\n")
print(responseFromChatbot.content)
print("\n----------\n")

# Let's now change the session_id and see what happens
session2 = {"configurable": {"session_id": "002"}}

responseFromChatbot = chatbot_with_message_history.invoke(
    [HumanMessage(content="What's my favorite color?")],
    config=session2,
)

print("\n----------\n")
print("What's my favorite color? (in session2)")
print("\n----------\n")
print(responseFromChatbot.content)
print("\n----------\n")

# Going back to session1 and see if the memory is still there
session1 = {"configurable": {"session_id": "001"}}

responseFromChatbot = chatbot_with_message_history.invoke(
    [HumanMessage(content="What's my favorite color?")],
    config=session1,
)

print("\n----------\n")
print("What's my favorite color? (in session1 again)")
print("\n----------\n")
print(responseFromChatbot.content)
print("\n----------\n")

# Our ChatBot has session memory now. Let's check if it remembers the conversation from session2.
session2 = {"configurable": {"session_id": "002"}}

responseFromChatbot = chatbot_with_message_history.invoke(
    [HumanMessage(content="Mi name is Julio.")],
    config=session2,
)

print("\n----------\n")
print("Mi name is Julio. (in session2)")
print("\n----------\n")
print(responseFromChatbot.content)
print("\n----------\n")

responseFromChatbot = chatbot_with_message_history.invoke(
    [HumanMessage(content="What is my name?")],
    config=session2,
)

print("\n----------\n")
print("What is my name? (in session2)")
print("\n----------\n")
print(responseFromChatbot.content)
print("\n----------\n")

responseFromChatbot = chatbot_with_message_history.invoke(
    [HumanMessage(content="What is my favorite color?")],
    config=session1,
)

print("\n----------\n")
print("What is my favorite color? (in session2)")
print("\n----------\n")
print(responseFromChatbot.content)
print("\n----------\n")




from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough

# To Define a function to limit the number of messages stored in memory and add it to our chain with .assign

def limited_memory_of_messages(messages, number_of_messages_to_keep=2):
    return messages[-number_of_messages_to_keep:]

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

limitedMemoryChain = (
    RunnablePassthrough.assign(messages=lambda x: limited_memory_of_messages(x["messages"]))
    | prompt 
    | chatbot
)

# To create our new chatbot with limited message history
chatbot_with_limited_message_history = RunnableWithMessageHistory(
    limitedMemoryChain,
    get_session_history,
    input_messages_key="messages",
)

# Adding 2 more messages to the session1 conversation
responseFromChatbot = chatbot_with_message_history.invoke(
    [HumanMessage(content="My favorite vehicles are Vespa scooters.")],
    config=session1,
)

print("\n----------\n")
print("My favorite vehicles are Vespa scooters. (in session1)")
print("\n----------\n")
print(responseFromChatbot.content)
print("\n----------\n")

responseFromChatbot = chatbot_with_message_history.invoke(
    [HumanMessage(content="My favorite city is San Francisco.")],
    config=session1,
)

print("\n----------\n")
print("My favorite city is San Francisco. (in session1)")
print("\n----------\n")
print(responseFromChatbot.content)
print("\n----------\n")

# The chatbot memory has now 4 messages. Let's check the Chatbot with limited memory.
responseFromChatbot = chatbot_with_limited_message_history.invoke(
    {
        "messages": [HumanMessage(content="what is my favorite color?")],
    },
    config=session1,
)

print("\n----------\n")
print("what is my favorite color? (chatbot with memory limited to the last 3 messages)")
print("\n----------\n")
print(responseFromChatbot.content)
print("\n----------\n")

# Finally, comparing the previous response with the one provided by the Chatbot with unlimited memory
responseFromChatbot = chatbot_with_message_history.invoke(
    [HumanMessage(content="what is my favorite color?")],
    config=session1,
)

print("\n----------\n")
print("what is my favorite color? (chatbot with unlimited memory)")
print("\n----------\n")
print(responseFromChatbot.content)
print("\n----------\n")