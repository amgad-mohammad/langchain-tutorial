# It is better to split the workflow to a smaller chains (easier for debugging,clearer and easier to modify)

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableLambda


def manual_invocation():
    user_input = input("Enter the framework name:")
    prompt_template = ChatPromptTemplate.from_messages(
        [("system", 'You are a helpful assistant for answering technology questions'),
         ("human", 'Who is the creator of {framework}?')  # we can use human or user
         ]
    )
    ollama_llm = ChatOllama(model='qwen2.5:3b', temperature=0)
    final_prompt_template = prompt_template.invoke({"framework": user_input})
    response= ollama_llm.invoke(final_prompt_template)
    print(response.content)


def chain_invocation_using_pipe_operator():
    # TASK-1
    user_input = input("Enter the framework name:")
    prompt_template = ChatPromptTemplate.from_messages(
        [("system", 'You are a helpful assistant for answering technology questions'),
         ("human", 'Who is the creator of {framework}?')  # we can use human or user
        ]
    )

    # TASK-2
    ollama_llm =ChatOllama(model='qwen2.5:3b',temperature=0)

    #Task-3
    #StrOutputParser : parses the output by extracting the content from it (without it, we do that by using .content)
    str_output_parser= StrOutputParser()

    # creating a chain using pipe(|) operator (LCEL:Lang chain Expression Language)
    chain= prompt_template | ollama_llm | str_output_parser
    #print(chain)
    llm_response= chain.invoke({'framework':user_input})
    print(llm_response)

def chain_invocation_using_runnable_sequence():
    # TASK-1
    user_input = input("Enter the framework name:")
    prompt_template = ChatPromptTemplate.from_messages(
        [("system", 'You are a helpful assistant for answering technology questions'),
         ("human", 'Who is the creator of {framework}?')  # we can use human or user
        ]
    )

    # TASK-2
    ollama_llm =ChatOllama(model='qwen2.5:3b',temperature=0)

    #Task-3
    #StrOutputParser : parses the output by extracting the content from it (without it, we do that by using .content)
    str_output_parser= StrOutputParser()

    # creating a chain using pipe(|) operator (LCEL:Lang chain Expression Language)
    chain= RunnableSequence(prompt_template,ollama_llm,str_output_parser)
    #print(chain)
    llm_response= chain.invoke({'framework':user_input})
    print(llm_response)


def chain_with_custom_runnable_invocation():
    # we want to create a function inside the chain that takes some input and transforms it into a desired output that matches with another input chain(custom Runnable , also called RunnableLambda)
    # TASK-1
    user_input = input("Enter the framework name:")
    prompt_template_1 = ChatPromptTemplate.from_messages(
        [("system", 'You are a helpful assistant for answering technology questions'),
         ("human", 'Who is the creator of {framework}?')  # we can use human or user
         ]
    )

    # TASK-2
    ollama_llm_1 = ChatOllama(model='qwen2.5:3b', temperature=0)

    # Task-3
    # StrOutputParser : parses the output by extracting the content from it (without it, we do that by using .content)
    str_output_parser_1 = StrOutputParser()

    # Task-4
    # dictionary Maker Lambda
    dictionary_maker_runnable = RunnableLambda(dictionary_maker)

    # Task-5
    # template for post
    prompt_template_2 = ChatPromptTemplate.from_messages(
        [("system", 'You are a social media post generator'),
         ("human", 'create a post for the following text on linkedIn: {text}')
         ]
    )

    # TASK-6
    ollama_llm_2 = ChatOllama(model='qwen2.5:3b', temperature=0)

    # Task-7
    str_output_parser_2 = StrOutputParser()

    # CORRECT CHAIN: removed the duplicate prompt_template_1 | ollama_llm_1 | str_output_parser_1
    final_chain = prompt_template_1 | ollama_llm_1 | str_output_parser_1 | dictionary_maker_runnable | prompt_template_2 | ollama_llm_2 | str_output_parser_2

    # CORRECT INVOCATION: use final_chain, not the intermediate chain
    llm_response = final_chain.invoke({'framework': user_input})

    print(llm_response)


def dictionary_maker(text: str) -> dict:
    return {"text": text}



if __name__ == '__main__':
    chain_invocation_using_runnable_sequence()