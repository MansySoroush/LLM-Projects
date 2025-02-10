# Sentiment Analysis App
"""
    A Text Classification App (Tagging App)
    To create an app to classify text into labels. And these labels can be:
        Sentiment: Sentiment Analysis app.
        Language: Language Analysis app.
        Style (formal, informal, etc): Style Analysis app.
        Topics or categories: Topic or category Analysis app.
        Political tendency: Political Analysis app.
"""

# Connect with the .env file located in the same directory of this file
import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
openai_api_key = os.environ["OPENAI_API_KEY"]

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

tagging_prompt = ChatPromptTemplate.from_template(
"""
    Extract the desired information from the following passage.

    Only extract the properties mentioned in the 'Classification' function.

    Passage:
    {input}
"""
)

# To define the 3 tags to analize in this app:
"""
    sentiment.
    political tendency.
    language. 
"""

# Defining our classification criteria
# A Pydantic model called Classification
class Classification(BaseModel):
    sentiment: str = Field(
        ..., 
        description="The sentiment of the text",
        enum=["happy", "neutral", "sad"])
    political_tendency: str = Field(
        ...,
        description="The political tendency of the user",
        enum=["conservative", "liberal", "independent"],
    )
    language: str = Field(
        ...,
        description="The language the text is written in", 
        enum=["spanish", "english"]
    )

# Connect with an LLM and start a conversation with it
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0125").with_structured_output(
    Classification
)

tagging_chain = tagging_prompt | llm

trump_follower = "I'm confident that President Trump's leadership and track record will once again resonate with Americans. His strong stance on economic growth and national security is exactly what our country needs at this pivotal moment. We need to bring back the proven leadership that can make America great again!"

biden_follower = "I believe President Biden's compassionate and steady approach is vital for our nation right now. His commitment to healthcare reform, climate change, and restoring our international alliances is crucial. It's time to continue the progress and ensure a future that benefits all Americans."

response = tagging_chain.invoke({"input": trump_follower})

print("\n----------\n")
print("Sentiment analysis Trump follower (with a list of options using enums):")
print("\n----------\n")
print(response)
print("\n----------\n")

response = tagging_chain.invoke({"input": biden_follower})

print("\n----------\n")
print("Sentiment analysis Biden follower (with a list of options using enums):")
print("\n----------\n")
print(response)
print("\n----------\n")