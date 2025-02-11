
# Q&A from SQL App
"""
    Creating a Q&A App over tabular data in databases.
    Asking a question about the data in a database in natural language.
    Getting back an answer also in natural language.
"""

# Connect with the .env file located in the same directory of this file
import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
openai_api_key = os.environ["OPENAI_API_KEY"]

# Connect with an LLM and start a conversation with it
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-3.5-turbo-0125")


# Connect with the database
"""
    Using a SQLite connection with the `street_tree_db` database in the `data` folder.
    Communicating with the database using the SQLAlchemy-driven `SQLDatabase` class.
"""

from langchain_community.utilities import SQLDatabase

sqlite_db_path = "data/street_tree_db.sqlite"

db = SQLDatabase.from_uri(f"sqlite:///{sqlite_db_path}")


# Step 1: Translating a question in natural language into an SQL query.
"""
    To take the user input and convert it to a SQL query.
    LangChain comes with a built-in chain for this, `create_sql_query_chain`
"""
from langchain.chains import create_sql_query_chain

write_query = create_sql_query_chain(llm, db)

response = write_query.invoke({"question": "List the species of trees that are present in San Francisco"})

print("\n----------\n")
print("List the species of trees that are present in San Francisco")
print("\n----------\n")
print(response)
print("\n----------\n")

# Executing the query to make sure it's valid
print("Query executed:")
print("\n----------\n")
print(db.run(response))
print("\n----------\n")


# To see how the pre-defined chain is built: calling get_prompts()
# How to solve the possible problem? or How to understand what is happening under the hook?
chain_prompt = write_query.get_prompts()[0].pretty_print()
print("Output of write_query.get_prompts():")
print("\n----------\n")
print(chain_prompt)
print("\n----------\n")


# Step 2: Executing the SQL query.
"""
    Using the QuerySQLDatabaseTool to easily add query execution to our chain:
        The user asks a question.
        The write_query chain has the question as input and the SQL query as output.
        The execute_query chain has the SQL query as input and the SQL query execution as output.
"""

from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool

execute_query = QuerySQLDataBaseTool(db=db)

temp_chain = write_query | execute_query

response = temp_chain.invoke({"question": "List the species of trees that are present in San Francisco"})

print("\n----------\n")
print("List the species of trees that are present in San Francisco (with query execution included)")
print("\n----------\n")
print(response)
print("\n----------\n")


# Step 3: Translate the SQL response into a natural language response
"""
    Combining the original question and SQL query result with the chat model to generate a final answer in natural language.
    Doing this by passing question and result to the LLM. (Using Prompt Template & RunnablePassthrough with .assign(...))

    Using RunnablePassthrough to get that "question" variable, and we use .assign() twice to get the other two variables required by the prompt template: 
        "query" and "result".

    With the first .assign():
        the write_query chain has the question as input and the SQL query (identified by the variable name "query") as output.

    With the second .assign():
        the execute_query chain has the SQL query as input and the SQL query execution (identified by the variable name "result") as output.

    The prompt template has:
        the question (identified by the variable name "question")
        the SQL query (identified by the variable name "query")
        the SQL query execution (identified by the variable name "result") as input
        the final prompt as the output.
"""

from operator import itemgetter

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

answer_prompt = PromptTemplate.from_template(
    """Given the following user question, corresponding SQL query, and SQL result, answer the user question.

    Question: {question}
    SQL Query: {query}
    SQL Result: {result}
    Answer: 
    """
)

chain = (
    RunnablePassthrough.assign(query=write_query).assign(
        result=itemgetter("query") | execute_query
    )
    | answer_prompt
    | llm
    | StrOutputParser()
)

response = chain.invoke({"question": "List the species of trees that are present in San Francisco"})

print("\n----------\n")
print("List the species of trees that are present in San Francisco (passing question and result to the LLM)")
print("\n----------\n")
print(response)
print("\n----------\n")
