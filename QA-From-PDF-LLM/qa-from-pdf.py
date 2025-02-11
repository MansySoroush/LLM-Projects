# Q&A from PDF App
"""
    Creating a Q&A App that answer questions about PDF files.
    Using a Document Loader to load text in a format usable by an LLM
    Building a retrieval-augmented generation (RAG) pipeline to answer questions
    Including citations from the source material.

    Using the basic approach for this project.
"""

# Connect with the .env file located in the same directory of this file
import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
openai_api_key = os.environ["OPENAI_API_KEY"]

# Connect with an LLM and start a conversation with it
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-3.5-turbo-0125")

# Load the PDF file
"""
    The loader reads the PDF at the specified path into memory.
    It then extracts text data using the pypdf package.
    Finally, it creates a LangChain Document for each page of the PDF with the page's content 
        and some metadata about where in the document the text came from.
"""

from langchain_community.document_loaders import PyPDFLoader

file_path = "./data/Be_Good.pdf"

loader = PyPDFLoader(file_path)

doc_pages = loader.load()

print("\n----------\n")
print("Content of the first page:")
print(doc_pages[0].page_content[0:100])
print("\n----------\n")
print("Metadata of the first page:")
print(doc_pages[0].metadata)
print("\n----------\n")

# Steps of the RAG process.
"""
Due to handling too large data to fit into the context window. the RAG technique will be used:
    Split document in small chunks.
    Transform text chunks in numeric chunks (embeddings).
    Load embeddings to a vector database (aka vector store) such as Chroma.
    Load question and retrieve the most relevant embeddings to respond it.
    Sent the embeddings to the LLM to format the response properly.
"""
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

chunks = text_splitter.split_documents(doc_pages)

vectorstore = Chroma.from_documents(documents=chunks, embedding=OpenAIEmbeddings())

retriever = vectorstore.as_retriever()


# Using two pre-defined chains to construct the final rag_chain:
#   create_stuff_documents_chain
#   create_retrieval_chain

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# create_stuff_documents_chain
"""
    It takes a list of documents and formats them all into a prompt.
    Then passes that prompt to an LLM. 
    Noted: It passes ALL documents, so you should make sure it fits within the context window of the LLM.
"""
question_answer_chain = create_stuff_documents_chain(llm, prompt)


# create_retrieval_chain
"""
    It takes in a user inquiry, which is then passed to the retriever to fetch relevant documents. 
    Those documents (and original inputs) are then passed to an LLM to generate a response.
"""
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

response = rag_chain.invoke({"input": "What is this article about?"})

print("\n----------\n")
print("What is this article about?")
print("\n----------\n")
print(response["answer"])
print("\n----------\n")

print("\n----------\n")
print("Show metadata of the first page:")
print("\n----------\n")
print(response["context"][0].metadata)
print("\n----------\n")