# Conversational RAG App
"""
    It allows the user to have a back-and-forth conversation
    It means that the application needs some sort of "memory" of past questions and answers.
"""

# Connect with the .env file located in the same directory of this file
import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
openai_api_key = os.environ["OPENAI_API_KEY"]

# Connect with an LLM and start a conversation with it
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-3.5-turbo-0125")

# Steps of the RAG process.
"""
Due to handling too large data to fit into the context window. the RAG technique will be used:
    Split document in small chunks.
    Transform text chunks in numeric chunks (embeddings).
    Load embeddings to a vector database (aka vector store) such as Chroma.
    Load question and retrieve the most relevant embeddings to respond it.
    Sent the embeddings to the LLM to format the response properly.
"""

# How do we handle when the user refers to previous Q&As in the conversation?
"""
    Store the chat conversation.
    When the user enters a new input, put that input in context.
    Re-phrase the user input to have a contextualized input.
    Send the contextualized input to the retriever.
    Use the retriever to build a conversational rag chain.
    Add extra features like persisting memory (save memory in a file) and session memories.
"""

# The process we will follow
"""
    1. Create a basic RAG without memory.
    2. Create a ChatPrompTemplate able to contextualize inputs.
    3. Create a retriever aware of memory.
    4. Create a basic conversational RAG.
    5. Create an advanced conversational RAG with persistence and session memories.
"""

import bs4
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Step 1: Create a basic RAG without memory

loader = TextLoader("./data/be-good.txt")

docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

splits = text_splitter.split_documents(docs)

vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

retriever = vectorstore.as_retriever()

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

# Step 2: Create a ChatPromptTemplate able to contextualize inputs
"""
    Goal: put the input in context and re-phrase it so we have a contextualized input.
    Defining a new system prompt that instructs the LLM in how to contextualize the input.
    Our new ChatPromptTemplate will include:
        The new system prompt.
        MessagesPlaceholder, a placeholder used to pass the list of messages included in the chat_history.
"""

from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder

contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Step 3: Create a Retriever aware of the memory
"""
    Building our new retriever with create_history_aware_retriever that uses the contextualized input to get a contextualized response.
"""

history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)


# Step 4: Create a basic Conversational RAG
"""
    Using the retriever aware of memory, that uses the prompt with contextualized input.
    Using create_stuff_documents_chain to build a qa chain:
        a chain able to asks questions to an LLM.
    Using create_retrieval_chain and the qa chain to build the RAG chain:
        a chain able to asks questions to the retriever and then format the response with the LLM.
"""

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


# Trying to test our basic Conversational RAG App
from langchain_core.messages import AIMessage, HumanMessage

chat_history = []

question = "What is this article about?"

ai_msg_1 = rag_chain.invoke({"input": question, "chat_history": chat_history})

chat_history.extend(
    [
        HumanMessage(content=question),
        AIMessage(content=ai_msg_1["answer"]),
    ]
)

second_question = "What was my previous question about?"

response = rag_chain.invoke({"input": second_question, "chat_history": chat_history})

print("\n----------\n")
print("What was my previous question about?")
print("\n----------\n")
print(response["answer"])
print("\n----------\n")


# Step 5: Advanced conversational RAG with persistence and session memories
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# To Create a dictionary to store our memory
memory_store = {}


# To Declare a function to restore and save the message by session id as the key (Separate memory for separate users)
# input: session_id, output: chatbotMemory[session_id]
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in memory_store:
        memory_store[session_id] = ChatMessageHistory()
    return memory_store[session_id]

# using `RunnableWithMessageHistory` to manage chat history with a configuration that includes a unique session identifier (`session_id`). 
# This identifier helps the system know which conversation history to retrieve and update whenever a user interacts with the system.
conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)


response = conversational_rag_chain.invoke(
    {"input": "What is this article about?"},
    config={
        "configurable": {"session_id": "001"}
    },  # constructs a key "001" in `store`.
)

print("\n----------\n")
print("What is this article about?")
print("\n----------\n")
print(response["answer"])
print("\n----------\n")

response = conversational_rag_chain.invoke(
    {"input": "What was my previous question about?"},
    config={"configurable": {"session_id": "001"}},
)

print("\n----------\n")
print("What was my previous question about?")
print("\n----------\n")
print(response["answer"])
print("\n----------\n")

print("\n----------\n")
print("Conversation History:")
print("\n----------\n")

for message in memory_store["001"].messages:
    if isinstance(message, AIMessage):
        prefix = "AI"
    else:
        prefix = "User"

    print(f"{prefix}: {message.content}\n")

print("\n----------\n")