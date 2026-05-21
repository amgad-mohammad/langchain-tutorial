from langchain_core.messages import HumanMessage,SystemMessage,AIMessage
from langchain_ollama import ChatOllama


def main():
    #SystemMessage : directs the LLM to a certain capability and context
    #HumanMessage : user message/question to LLM
    #AIMessage: LLM Response
    my_messages=[
        SystemMessage(content='You are a helpful assistant for answering technology questions'),
        HumanMessage(content='Who is the creator of langchain framework?')
    ]

    ollama_llm =ChatOllama(model='qwen2.5:3b',temperature=0)
    ai_message= ollama_llm.invoke(my_messages) #AIMessage
    print(ai_message.content)

if __name__ == '__main__':
    main()