import streamlit as st
from langchain import PromptTemplate
from langchain_openai import OpenAI

import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
openai_api_key = os.environ["OPENAI_API_KEY"]

#Page title and header
st.set_page_config(page_title = "Blog Post Generator")
st.title("Blog Post Generator")

st.markdown("Please, Enter the Topic first.")

def generate_response(topic):
    llm = OpenAI(openai_api_key=openai_api_key)
    template = """
        As experienced startup and venture capital writer, 
        generate a 400-word blog post about {topic}
        
        Your response should be in this format:
        First, print the blog post.
        Then, sum the total number of words on it and print the result like this: This post has X words.
    """
    prompt = PromptTemplate(
        input_variables = ["topic"],
        template = template
    )
    query = prompt.format(topic=topic)
    response = llm(query, max_tokens=2048)
    return st.write(response)

topic_text = st.sidebar.text_input("Enter the topic: ")
if topic_text:
    generate_response(topic_text)
