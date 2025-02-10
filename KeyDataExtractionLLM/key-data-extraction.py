# Key Data Extraction App
"""
    To extract structured information from unstructured text.
    e.g. To to extract the name, the lastname and the country of the users that submit comments in the website of your company.
"""

# Connect with the .env file located in the same directory of this file
import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
openai_api_key = os.environ["OPENAI_API_KEY"]

# Connect with an LLM and start a conversation with it
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-3.5-turbo-0125")


from typing import List, Optional

from langchain_core.pydantic_v1 import BaseModel, Field

# Define what information you want to extract
class User(BaseModel):
    """Information about a User."""

    # ^ Doc-string for the entity User.
    # This doc-string is sent to the LLM as the description of the schema User,
    # and it can help to improve extraction results.

    # Note that:
    # 1. Each field is an `optional` -- this allows the model to decline to extract it!
    # 2. Each field has a `description` -- this description is used by the LLM.
    # Having a good description can help improve extraction results.
    name: Optional[str] = Field(default=None, description="The name of the User")
    lastname: Optional[str] = Field(
        default=None, description="The lastname of the User if known"
    )
    country: Optional[str] = Field(
        default=None, description="The country of the User if known"
    )

class Users(BaseModel):
    """Extracted data about people."""

    # Creates a model so that we can extract multiple entities.
    people: List[User]
    

from langchain_core.prompts import ChatPromptTemplate

# Define a custom prompt to provide instructions and any additional context.
"""
    1) You can add examples into the prompt template to improve extraction quality
    2) Introduce additional parameters to take context into account (e.g., include metadata
        about the document from which the text was extracted.)
"""

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert extraction algorithm. "
            "Only extract relevant information from the text. "
            "If you do not know the value of an attribute asked to extract, "
            "return null for the attribute's value.",
        ),
        ("human", "{text}"),
    ]
)

# Define the Extractor
# Our extractor will be a chain with the prompt template and a chat model with the extraction instructions.
chain = prompt | llm.with_structured_output(schema=Users)

comment = "I'm so impressed with this product! It has truly transformed how I approach my daily tasks. The quality exceeds my expectations, and the customer support is truly exceptional. I've already suggested it to all my colleagues and relatives. - Emily Clarke, Canada"

response = chain.invoke({"text": comment})

print("\n----------\n")
print("Key data extraction of a list of entities:")
print("\n----------\n")
print(response)
print("\n----------\n")

# Example input text that mentions multiple people
text_input = """
Alice Johnson from Canada recently reviewed a book she loved. Meanwhile, Bob Smith from the USA shared his insights on the same book in a different review. Both reviews were very insightful.
"""

# Invoke the processing chain on the text
response = chain.invoke({"text": text_input})

# Output the extracted data
print("\n----------\n")
print("Key data extraction of a review with several users:")
print("\n----------\n")
print(response)
print("\n----------\n")
