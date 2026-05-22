"""
LANGCHAIN CHAINS - COMPLETE GUIDE
=================================
Learn how to build and chain LLM operations step by step.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableLambda, RunnableParallel, RunnableBranch


# ============================================================================
# LEVEL 1: MANUAL INVOCATION (The Hard Way)
# ============================================================================
def level_1_manual_invocation():
    """
    HOW IT WORKS: Call each component manually, one by one.
    WHEN TO USE: Learning basics, debugging specific steps.
    DOWNSIDE: Verbose, repetitive code.
    """
    user_input = input("Enter framework name: ")

    # Step 1: Create prompt
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful tech assistant"),
        ("human", "Who created {framework}?")
    ])

    # Step 2: Fill in the prompt
    final_prompt = prompt_template.invoke({"framework": user_input})

    # Step 3: Call LLM
    llm = ChatOllama(model='qwen2.5:3b', temperature=0)
    response = llm.invoke(final_prompt)

    # Step 4: Extract content manually
    print(response.content)  # Notice we need .content


# ============================================================================
# LEVEL 2: CHAIN WITH PIPE OPERATOR (The Easy Way) ⭐ RECOMMENDED
# ============================================================================
def level_2_chain_with_pipe():
    """
    HOW IT WORKS: Use | (pipe) to connect components automatically.
    WHEN TO USE: 99% of the time - this is the standard way!
    BENEFITS: Clean, readable, automatic data flow.

    FLOW: prompt → LLM → parser → result
    """
    user_input = input("Enter framework name: ")

    # Define components
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful tech assistant"),
        ("human", "Who created {framework}?")
    ])
    llm = ChatOllama(model='qwen2.5:3b', temperature=0)
    parser = StrOutputParser()  # Automatically extracts .content

    # Chain them with | operator
    chain = prompt | llm | parser

    # One invoke, automatic flow!
    result = chain.invoke({"framework": user_input})
    print(result)  # No .content needed - parser handles it!


# ============================================================================
# LEVEL 3: CHAIN WITH RUNNABLE SEQUENCE (Alternative Syntax)
# ============================================================================
def level_3_chain_with_runnable_sequence():
    """
    HOW IT WORKS: Same as pipe, but using RunnableSequence() explicitly.
    WHEN TO USE: When you prefer explicit function calls over operators.
    NOTE: This is the same as using |, just different syntax.
    """
    user_input = input("Enter framework name: ")

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful tech assistant"),
        ("human", "Who created {framework}?")
    ])
    llm = ChatOllama(model='qwen2.5:3b', temperature=0)
    parser = StrOutputParser()

    # Using RunnableSequence instead of |
    chain = RunnableSequence(prompt, llm, parser)

    result = chain.invoke({"framework": user_input})
    print(result)


# ============================================================================
# LEVEL 4: CUSTOM RUNNABLE (Adding Your Own Functions)
# ============================================================================
def level_4_custom_runnable():
    """
    HOW IT WORKS: Add custom Python functions into the chain.
    WHEN TO USE: Need to transform data between chain steps.

    EXAMPLE: Get creator name → convert to dict → generate LinkedIn post
    FLOW: prompt → LLM → parser → custom_function → prompt2 → LLM2 → parser2
    """
    user_input = input("Enter framework name: ")

    # CHAIN PART 1: Get creator name
    prompt_1 = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful tech assistant"),
        ("human", "Who created {framework}?")
    ])
    llm_1 = ChatOllama(model='qwen2.5:3b', temperature=0)
    parser_1 = StrOutputParser()

    # CUSTOM FUNCTION: Convert string → dict (needed for next prompt)
    def text_to_dict(text: str) -> dict:
        """Transforms: 'Jordan Walke' → {'text': 'Jordan Walke'}"""
        return {"text": text}

    custom_runnable = RunnableLambda(text_to_dict)

    # CHAIN PART 2: Generate LinkedIn post
    prompt_2 = ChatPromptTemplate.from_messages([
        ("system", "You are a LinkedIn post writer"),
        ("human", "Write a LinkedIn post about: {text}")
    ])
    llm_2 = ChatOllama(model='qwen2.5:3b', temperature=0)
    parser_2 = StrOutputParser()

    # COMPLETE CHAIN
    full_chain = (
        prompt_1 | llm_1 | parser_1 |      # Get creator
        custom_runnable |                   # Convert format
        prompt_2 | llm_2 | parser_2         # Generate post
    )

    result = full_chain.invoke({"framework": user_input})
    print("\n📝 LinkedIn Post:")
    print(result)


# ============================================================================
# LEVEL 5: PARALLEL CHAINS (Run Multiple Chains at Once)
# ============================================================================
def level_5_parallel_chains():
    """
    HOW IT WORKS: Run multiple chains simultaneously with same input.
    WHEN TO USE: Generate different outputs from same source (LinkedIn + Instagram posts).

    FLOW:
                    → LinkedIn chain → LinkedIn post
    summary text →
                    → Instagram chain → Instagram post
    """
    user_input = input("Enter movie name: ")

    # STEP 1: Summarize the movie
    summary_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a movie expert"),
        ("human", "Summarize the movie '{movie}' in 2 sentences")
    ])
    llm = ChatOllama(model='qwen2.5:3b', temperature=0)
    parser = StrOutputParser()

    # Custom function to convert summary to dict
    def text_to_dict(text: str) -> dict:
        return {"text": text}

    # STEP 2: Create LinkedIn chain
    linkedin_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a professional LinkedIn writer"),
        ("human", "Create a LinkedIn post about: {text}")
    ])
    linkedin_chain = linkedin_prompt | llm | parser

    # STEP 3: Create Instagram chain
    instagram_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a casual Instagram writer"),
        ("human", "Create an Instagram caption about: {text}")
    ])
    instagram_chain = instagram_prompt | llm | parser

    # STEP 4: Combine with RunnableParallel
    full_chain = (
        summary_prompt | llm | parser |
        RunnableLambda(text_to_dict) |
        RunnableParallel(
            linkedin=linkedin_chain,
            instagram=instagram_chain
        )
    )

    # Execute - both chains run in parallel!
    result = full_chain.invoke({"movie": user_input})

    print("\n📘 LinkedIn Post:")
    print(result["linkedin"])
    print("\n📸 Instagram Caption:")
    print(result["instagram"])


# ============================================================================
# LEVEL 6: CHAIN AS A RUNNABLE (Reusing Chains)
# ============================================================================
def level_6_chain_as_runnable():
    """
    HOW IT WORKS: Treat an entire chain as a single reusable component.
    WHEN TO USE: Build complex workflows from smaller, tested chains.

    CONCEPT: A chain IS a runnable, so you can pipe it like any other component!
    """
    user_input = input("Enter movie name: ")

    # Build a reusable summary chain
    summary_chain = (
        ChatPromptTemplate.from_messages([
            ("system", "You are a movie expert"),
            ("human", "Summarize '{movie}' in one sentence")
        ]) |
        ChatOllama(model='qwen2.5:3b', temperature=0) |
        StrOutputParser()
    )

    # Use the entire summary_chain AS A COMPONENT in a bigger chain
    def text_to_dict(text: str) -> dict:
        return {"text": text}

    post_chain = (
        ChatPromptTemplate.from_messages([
            ("system", "You are a social media expert"),
            ("human", "Make this exciting: {text}")
        ]) |
        ChatOllama(model='qwen2.5:3b', temperature=0) |
        StrOutputParser()
    )

    # Combine: summary_chain is just another runnable!
    full_chain = (
        summary_chain |              # Entire chain used as one step
        RunnableLambda(text_to_dict) |
        post_chain
    )

    result = full_chain.invoke({"movie": user_input})
    print("\n✨ Final Post:")
    print(result)


# ============================================================================
# LEVEL 7: CONDITIONAL CHAINS (If-Else Logic in Chains)
# ============================================================================
def level_7_conditional_chains_basic():
    """
    HOW IT WORKS: Route to different chains based on conditions (like if-else).
    WHEN TO USE: Different processing based on input type, user role, content category.

    EXAMPLE: Route question to different experts based on topic.

    FLOW:
                  → Technical chain (if tech question)
    question →    → Business chain (if business question)
                  → General chain (default)
    """
    user_input = input("Ask a question: ")

    llm = ChatOllama(model='qwen2.5:3b', temperature=0)
    parser = StrOutputParser()

    # OPTION 1: Technical expert chain
    tech_chain = (
        ChatPromptTemplate.from_messages([
            ("system", "You are a senior software engineer. Give detailed technical answers."),
            ("human", "{question}")
        ]) | llm | parser
    )

    # OPTION 2: Business expert chain
    business_chain = (
        ChatPromptTemplate.from_messages([
            ("system", "You are a business consultant. Focus on ROI and business value."),
            ("human", "{question}")
        ]) | llm | parser
    )

    # OPTION 3: General chain (default)
    general_chain = (
        ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant."),
            ("human", "{question}")
        ]) | llm | parser
    )

    # CONDITION FUNCTIONS: Return True if this branch should run
    def is_technical(input_dict: dict) -> bool:
        """Check if question is technical"""
        question = input_dict["question"].lower()
        tech_keywords = ["code", "programming", "api", "database", "python", "javascript"]
        return any(keyword in question for keyword in tech_keywords)

    def is_business(input_dict: dict) -> bool:
        """Check if question is business-related"""
        question = input_dict["question"].lower()
        business_keywords = ["revenue", "profit", "market", "sales", "business", "roi"]
        return any(keyword in question for keyword in business_keywords)

    # BUILD CONDITIONAL CHAIN
    # RunnableBranch format: (condition_function, chain_to_run)
    conditional_chain = RunnableBranch(
        (is_technical, tech_chain),      # If technical → tech expert
        (is_business, business_chain),   # Else if business → business expert
        general_chain                     # Else → general assistant (default, no condition)
    )

    result = conditional_chain.invoke({"question": user_input})
    print("\n💡 Answer:")
    print(result)


def level_7_conditional_chains_advanced():
    """
    ADVANCED: Conditional chain with sentiment analysis routing.

    FLOW:
                  → Positive response chain (if positive sentiment)
    feedback →    → Negative response chain (if negative sentiment)
                  → Neutral response chain (default)
    """
    user_feedback = input("Give us feedback: ")

    llm = ChatOllama(model='qwen2.5:3b', temperature=0)
    parser = StrOutputParser()

    # Chain 1: Positive feedback response
    positive_chain = (
        ChatPromptTemplate.from_messages([
            ("system", "You are a grateful customer service agent. Thank the user warmly."),
            ("human", "User feedback: {feedback}")
        ]) | llm | parser
    )

    # Chain 2: Negative feedback response
    negative_chain = (
        ChatPromptTemplate.from_messages([
            ("system", "You are an empathetic customer service agent. Apologize and offer solutions."),
            ("human", "User feedback: {feedback}")
        ]) | llm | parser
    )

    # Chain 3: Neutral feedback response
    neutral_chain = (
        ChatPromptTemplate.from_messages([
            ("system", "You are a professional customer service agent. Acknowledge and ask for details."),
            ("human", "User feedback: {feedback}")
        ]) | llm | parser
    )

    # Sentiment detection functions
    def is_positive(input_dict: dict) -> bool:
        feedback = input_dict["feedback"].lower()
        positive_words = ["great", "excellent", "love", "amazing", "fantastic", "good", "wonderful"]
        return any(word in feedback for word in positive_words)

    def is_negative(input_dict: dict) -> bool:
        feedback = input_dict["feedback"].lower()
        negative_words = ["bad", "terrible", "hate", "awful", "poor", "worst", "horrible"]
        return any(word in feedback for word in negative_words)

    # Conditional routing
    sentiment_router = RunnableBranch(
        (is_positive, positive_chain),
        (is_negative, negative_chain),
        neutral_chain  # Default
    )

    result = sentiment_router.invoke({"feedback": user_feedback})
    print("\n📧 Response:")
    print(result)


def level_7_conditional_chains_with_lambda():
    """
    ALTERNATIVE: Using RunnableLambda for conditional logic.
    WHEN TO USE: When you need more complex routing logic.
    """
    user_input = input("Enter a number (1-100): ")

    llm = ChatOllama(model='qwen2.5:3b', temperature=0)
    parser = StrOutputParser()

    # Different chains based on number range
    small_chain = (
        ChatPromptTemplate.from_messages([
            ("system", "You are enthusiastic about small numbers."),
            ("human", "Tell me something cool about the number {number}")
        ]) | llm | parser
    )

    medium_chain = (
        ChatPromptTemplate.from_messages([
            ("system", "You are analytical about medium numbers."),
            ("human", "Analyze the number {number} mathematically")
        ]) | llm | parser
    )

    large_chain = (
        ChatPromptTemplate.from_messages([
            ("system", "You are impressed by large numbers."),
            ("human", "Explain why {number} is a significant number")
        ]) | llm | parser
    )

    # Custom routing function
    def route_by_number(input_dict: dict) -> str:
        """Returns the chain result based on number value"""
        number = int(input_dict["number"])

        if number < 30:
            return small_chain.invoke(input_dict)
        elif number < 70:
            return medium_chain.invoke(input_dict)
        else:
            return large_chain.invoke(input_dict)

    # Wrap in RunnableLambda
    router = RunnableLambda(route_by_number)

    result = router.invoke({"number": user_input})
    print("\n🔢 Result:")
    print(result)


# ============================================================================
# QUICK REFERENCE
# ============================================================================
"""
WHEN TO USE EACH LEVEL:

✅ Level 2 (Pipe Operator): USE THIS 90% of the time
   - Clean syntax
   - Easy to read
   - Standard LangChain pattern

✅ Level 4 (Custom Runnable): When you need custom logic
   - Data transformation
   - API calls
   - Custom processing

✅ Level 5 (Parallel): When you need multiple outputs from same input
   - Different social media posts
   - Multiple translations
   - Various formats

✅ Level 7 (Conditional): When you need different processing based on input
   - Route by topic/category
   - Sentiment-based responses
   - User role-based chains
   - A/B testing different prompts

❌ Level 1 (Manual): Only for learning/debugging
❌ Level 3 (RunnableSequence): Only if you dislike | operator

CONDITIONAL CHAINS PATTERNS:

1. RunnableBranch (Recommended):
   - Clean if-else-if-else structure
   - Multiple conditions with default
   - Format: RunnableBranch((condition, chain), (condition, chain), default_chain)

2. RunnableLambda (Flexible):
   - Complex routing logic
   - Multiple return paths
   - Full Python control
"""


# ============================================================================
# RUN EXAMPLES
# ============================================================================
if __name__ == '__main__':
    print("="*60)
    print("LANGCHAIN CHAINS - PROGRESSIVE LEARNING")
    print("="*60)

    # Uncomment the level you want to try:

    level_1_manual_invocation()              # The hard way
    # level_2_chain_with_pipe()                # ⭐ START HERE
    # level_3_chain_with_runnable_sequence()   # Alternative syntax
    # level_4_custom_runnable()                # Adding custom functions
    # level_5_parallel_chains()                # Multiple outputs
    # level_6_chain_as_runnable()              # Reusing chains

    # NEW: Conditional Chains
    level_7_conditional_chains_basic()       # If-else routing
    # level_7_conditional_chains_advanced()    # Sentiment routing
    # level_7_conditional_chains_with_lambda() # Custom routing logic