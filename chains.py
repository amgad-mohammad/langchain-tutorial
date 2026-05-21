# It is better to split the workflow to a smaller chains (easier for debugging,clearer and easier to modify)

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence


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

if __name__ == '__main__':
    chain_invocation_using_runnable_sequence()