import streamlit as st
from langchain import PromptTemplate
from langchain_openai import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from io import StringIO

import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
openai_api_key = os.environ["OPENAI_API_KEY"]

#LLM and key loading function
def load_LLM():
    """Logic for loading the chain you want to use should go here."""
    # Make sure your openai_api_key is set as an environment variable
    llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
    return llm


#Page title and header
st.set_page_config(page_title="AI Long Text Summarizer")
st.header("AI Long Text Summarizer")

# Input
st.markdown("## Upload the text file you want to summarize")

uploaded_file = st.file_uploader("Choose a file", type="txt")

# Output
st.markdown("### Here is your Summary:")

if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()

    # To convert to a string based IO:
    string_io = StringIO(uploaded_file.getvalue().decode("utf-8"))

    # To read file as string:
    string_data = string_io.read()
    #st.write(string_data)

    file_input = string_data

    if len(file_input.split(" ")) > 20000:
        st.write("Please enter a shorter file. The maximum length is 20000 words.")
        st.stop()

    if file_input:
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n"], 
            chunk_size=5000, 
            chunk_overlap=350
        )

        splitted_documents = text_splitter.create_documents([file_input])

        llm = load_LLM(openai_api_key=openai_api_key)

        summarize_chain = load_summarize_chain(
            llm=llm, 
            chain_type="map_reduce"
            )

        summary_output = summarize_chain.run(splitted_documents)

        st.write(summary_output)
