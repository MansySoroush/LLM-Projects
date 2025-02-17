import streamlit as st
from langchain_openai import OpenAI
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain

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

def generate_response(txt):
    llm = load_LLM(openai_api_key=openai_api_key)

    text_splitter = CharacterTextSplitter()
    texts = text_splitter.split_text(txt)
    docs = [Document(page_content=t) for t in texts]
    chain = load_summarize_chain(
        llm,
        chain_type="map_reduce"
    )
    return chain.run(docs)

#Page title and header
st.set_page_config(page_title="Writing Text Summarization")
st.header("Writing Text Summarization")

result = []
with st.form("summarize_form", clear_on_submit=False):
    txt_input = st.text_area(
        "Enter your text",
        "",
        height=200
    )

    submitted = st.form_submit_button("Summurize")
    if submitted:
        response = generate_response(txt_input)
        result.append(response)

if len(result):
    st.info(response)