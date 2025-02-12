# Simple RAG App
"""
    It's like a smart assistant that not only finds the information you need from a large document,
    but also uses that information to create new, useful responses.
"""

# Connect with the .env file located in the same directory of this file
import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
openai_api_key = os.environ["OPENAI_API_KEY"]

# Connect with an LLM and start a conversation with it
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-3.5-turbo")

# Steps of the RAG process.
"""
Due to handling too large data to fit into the context window. the RAG technique will be used:
    Split document in small chunks.
    Transform text chunks in numeric chunks (embeddings).
    Load embeddings to a vector database (aka vector store) such as Chroma.
    Load question and retrieve the most relevant embeddings to respond it.
    Sent the embeddings to the LLM to format the response properly.
"""

import bs4
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import HumanMessagePromptTemplate
from langchain_core.prompts import PromptTemplate

loader = TextLoader("./data/be-good.txt")

docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

chunks = text_splitter.split_documents(docs)

vectorstore = Chroma.from_documents(documents=chunks, embedding=OpenAIEmbeddings())

retriever = vectorstore.as_retriever()

#the following line is not compatible with python 3.11.4
#to install langchain-hub, you will have to use python 3.12.2 or superior

#from langchain import hub
#prompt = hub.pull("rlm/rag-prompt")

#to keep using python 3.11.4, we will paste the prompt from the hub
prompt  = ChatPromptTemplate(input_variables=['context', 'question'], metadata={'lc_hub_owner': 'rlm', 'lc_hub_repo': 'rag-prompt', 'lc_hub_commit_hash': '50442af133e61576e74536c6556cefe1fac147cad032f4377b60c436e6cdcb6e'}, messages=[HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['context', 'question'], template="You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\nQuestion: {question} \nContext: {context} \nAnswer:"))])

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

response = rag_chain.invoke("What is this article about?")

print("\n----------\n")
print("What is this article about?")
print("\n----------\n")
print(response)
print("\n----------\n")