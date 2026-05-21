
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel,Field

def enforce_output_format():

    user_input = input("Enter the framework name:")
    # this is finally converted to System & Human Messages
    #############Goal############ : make sure we always get the same response format
    # we direct the LLM to return the result in a specific format (return the result in this format key)
    #it is easy here,but in case the prompt message is too large and complex it will not easy and trusted to direct LLM
    #############Solution############ : using Pydantic Models

    prompt_template = ChatPromptTemplate.from_messages(
        [("system", 'You are a helpful assistant for answering technology questions'),
         ("user", 'Who is the creator of {framework}? return the result in this format key: the framework name and value: the creator name'),
         ]
    )
    final_prompt_template = prompt_template.invoke({"framework": user_input})
    ollama_llm = ChatOllama(model='qwen2.5:3b', temperature=0)
    llm_response = ollama_llm.invoke(final_prompt_template)
    print(llm_response.content)


    ######Structured Output(schema) using Pydantic Library#########

class LLMSchema(BaseModel):
      framework_name: str = Field(description="This is the Framework name the user asks about")
      creator_name: str = Field(description="This is the Creator name who created the framework")
    # to use this llm_schema, you have to pass these exact values with the defined data types otherwise, the LLM will not be able to create object of this class
    # feed the Schema with a description also to help it

def enforce_pydantic_schema():

    user_input = input("Enter the framework name:")
    # this is finally converted to System & Human Messages
    #############Goal############ : make sure we always get the same response format
    # we direct the LLM to return the result in a specific format (return the result in this format key)
    #it is easy here,but in case the prompt message is too large and complex it will not easy and trusted to direct LLM
    #############Solution############ : using Pydantic Models through pydantic library

    prompt_template = ChatPromptTemplate.from_messages(
        [("system", 'You are a helpful assistant for answering technology questions'),
         ("user", 'Who is the creator of {framework}?'),
         ]
    )
    final_prompt_template = prompt_template.invoke({"framework": user_input})
    ollama_llm = ChatOllama(model='qwen2.5:3b', temperature=0)
    llm_structured_response= ollama_llm.with_structured_output(LLMSchema)
    llm_response = llm_structured_response.invoke(final_prompt_template) #llm_response is of type LLMSchema, we can use its fields
    print(llm_response.creator_name)
    print(llm_response)

if __name__ == '__main__':
    enforce_pydantic_schema()