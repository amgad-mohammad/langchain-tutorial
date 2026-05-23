import sqlite3
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_ollama import ChatOllama


# ============================================================================
# STEP 1: CREATE YOUR CUSTOM TOOL WITH DETAILED DOCSTRING
# ============================================================================
@tool
def execute_sql_query(query: str) -> str:
    """
    Execute SQL SELECT query on the sales database.

    This tool runs queries on a database with an 'orders' table containing:
    - id: unique order identifier
    - customer_name: name of the customer
    - product_name: name of the product (e.g., 'Laptop', 'Tablet', 'Smartphone')
    - quantity: number of items purchased
    - price: price per item
    - total: total cost (quantity × price)

    Args:
        query: A valid SQL SELECT query

    Examples:
        - "SELECT * FROM orders"
        - "SELECT SUM(total) FROM orders WHERE product_name='Tablet'"
        - "SELECT customer_name, product_name FROM orders ORDER BY total DESC LIMIT 5"

    Returns:
        Query results as formatted list of dictionaries

    Note: Only SELECT queries are allowed for safety.
    """
    # Validate query type
    if not query.strip().upper().startswith("SELECT"):
        return "Error: Only SELECT queries are allowed for safety."

    try:
        conn = sqlite3.connect("sales_db.db")
        cursor = conn.cursor()

        # Execute query
        cursor.execute(query)
        results = cursor.fetchall()

        # Get column names
        columns = [desc[0] for desc in cursor.description]

        conn.close()

        if not results:
            return "No results found."

        # Format as list of dictionaries for readability
        formatted = []
        for row in results:
            formatted.append(dict(zip(columns, row)))

        return str(formatted)

    except sqlite3.Error as e:
        return f"SQL Error: {str(e)}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"


# ============================================================================
# STEP 2: INITIALIZE MODEL
# ============================================================================
llm = ChatOllama(
    model="qwen2.5:3b",
    temperature=0
)

# ============================================================================
# STEP 3: CREATE AGENT (Modern Way - 2026)
# ============================================================================
agent = create_agent(
    model=llm,
    tools=[execute_sql_query],
    system_prompt="""You are a helpful SQL assistant with access to a sales database.

When answering questions:
1. Analyze what data the user needs
2. Generate an appropriate SQL SELECT query
3. Use the execute_sql_query tool to run it
4. Interpret the results clearly for the user

The database has an 'orders' table with customer purchases.
Always use proper SQL syntax and be specific with queries."""
)

# ============================================================================
# STEP 4: INVOKE THE AGENT
# ============================================================================
if __name__ == '__main__':
    # Example questions
    questions = [
        "Show me the top 5 highest orders",
        "How much total sales did we make for Tablet?",
        "Who bought a Laptop?",
        "What is our total revenue across all products?"
    ]

    # Test with first question
    question = questions[0]

    print(f"\n{'=' * 60}")
    print(f"Question: {question}")
    print(f"{'=' * 60}\n")

    try:
        response = agent.invoke({
            "messages": [
                {"role": "user", "content": question}
            ]
        })

        # Get the final answer
        final_message = response["messages"][-1]

        print(f"\n{'=' * 60}")
        print("FINAL ANSWER:")
        print(f"{'=' * 60}")
        print(final_message.content)

    except Exception as e:
        print(f"Error: {e}")