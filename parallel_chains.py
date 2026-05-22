# It is better to split the workflow to a smaller chains (easier for debugging, clearer and easier to modify)

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda

# TASK -1 [Prompt]
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a movie summarizer"),
    ("human", "Please summarize the movie in brief : {input}")])

# TASK - 2 [LLM]
llm_openai = ChatOllama(model="qwen2.5:3b", temperature=0)

# TASK - 3 [Str Parser]
str_parser = StrOutputParser()


# TASK - 4 [Custom Runnable]
def dictionary_maker(text: str) -> dict:
    return {"text": text}


dictionary_maker_runnable = RunnableLambda(dictionary_maker)


# LinkedIn Chain
def build_linkedin_chain():
    linkedin_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a LinkedIn post generator"),
        ("human", "Create a post for the following text for LinkedIn: {text}")])

    llm_openai_1 = ChatOllama(model="qwen2.5:3b", temperature=0)
    str_parser_1 = StrOutputParser()

    return linkedin_prompt | llm_openai_1 | str_parser_1


# Instagram Chain
def insta_chain(text: dict):
    text_content = text["text"]

    insta_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an Instagram post generator"),
        ("human", "Create a post for the following text for Instagram: {text}")])

    llm_openai_insta = ChatOllama(model="qwen2.5:3b", temperature=0)
    str_parser_insta = StrOutputParser()

    chain_insta = insta_prompt | llm_openai_insta | str_parser_insta

    result = chain_insta.invoke({"text": text_content})

    return result


insta_chain_runnable = RunnableLambda(insta_chain)

# Build the chains
chain_linkedin = build_linkedin_chain()

# Final chain
final_chain = (
    prompt_template |
    llm_openai |
    str_parser |
    dictionary_maker_runnable |
    RunnableParallel(linkedin=chain_linkedin, instagram=insta_chain_runnable)
)


def parallel_chains():
    user_input = input("Enter the movie name: ")
    result = final_chain.invoke({"input": user_input})

    print("\n=== LinkedIn Post ===")
    print(result["linkedin"])

    print("\n=== Instagram Post ===")
    print(result["instagram"])


if __name__ == '__main__':
    parallel_chains()