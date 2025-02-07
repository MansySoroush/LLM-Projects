import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
openai_api_key = os.environ["OPENAI_API_KEY"]

from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-3.5-turbo-0125")

import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings

# Loading and Processing the Web Content:
"""
Using BeautifulSoup to extract only relevant parts:
    "post-content" → Main text of the blog.
    "post-title" → Title of the post.
    "post-header" → Header content.
"""

loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)

docs = loader.load()

# Splitting Text into Chunks
"""
Breaking the document into 1000-character chunks with 200-character overlap to ensure context continuity.
Splitting the webpage content into manageable text chunks.
"""

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

splits = text_splitter.split_documents(docs)

# Creating a Vector Store
"""
Converting text chunks into embeddings
Storing the embeddings in ChromaDB for fast retrieval
"""
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

# Converting the vector database into a retriever
retriever = vectorstore.as_retriever()

# Defining a Prompt and Formatting Function
"""
Fetching a predefined RAG prompt template from LangChain Hub.
"""
prompt = hub.pull("rlm/rag-prompt")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Creating a RAG Pipeline
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

# Running the RAG Chain
response = rag_chain.invoke("What is Task Decomposition?")

print("\n----------\n")
print("Response:")
print("\n----------\n")
print(response)
print("\n----------\n")