import ollama
from openai import OpenAI
from langchain_ollama import ChatOllama
import os

def main():
    llm_call_using_langchain_ollama_sdk()

def llm_call_using_ollama_sdk():
    response =ollama.chat(model='qwen2.5:3b',messages=[{
    'role': 'user',
    'content': 'Why is the sky blue?',}])
    print(response['message']['content'])

def llm_call_using_openai_sdk():
    openAIAPIKey = os.environ.get('OPENAI_API_KEY')
    # The client looks for an OPENAI_API_KEY environment variable by default(optional to pass it)
    client = OpenAI(apiKey=openAIAPIKey)
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Why is the sky blue?"}
        ]
    )
    print(completion.choices[0].message.content)


def llm_call_using_langchain_ollama_sdk():
    ollama_llm=ChatOllama(model='qwen2.5:3b',temperature=0)
    response= ollama_llm.invoke('Why is the sky blue?') #AIMessage
    print(response.content)


if __name__ == '__main__':
    main()