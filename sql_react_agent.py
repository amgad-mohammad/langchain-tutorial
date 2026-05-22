from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_ollama import ChatOllama

llm = ChatOllama(
    model='qwen2.5:3b',
    temperature=0,
    timeout=60  # Add timeout
)

sql_db = SQLDatabase.from_uri("sqlite:///sales_db.db")

agent = create_sql_agent(
    llm=llm,
    db=sql_db,
    verbose=True,
    agent_type="zero-shot-react-description",
    max_iterations=3,  # Limit iterations
    max_execution_time=120  # Add execution timeout
)

example_query = "How much total sales we made for Tablet"

try:
    response = agent.invoke({"input": example_query})
    print("\n=== Final Answer ===")
    print(response["output"])
except Exception as e:
    print(f"\nError occurred: {e}")