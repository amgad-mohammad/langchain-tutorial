import ollama
from openai import OpenAI
from langchain_ollama import ChatOllama
from langchain.chat_models import init_chat_model
import os
from dotenv import load_dotenv

def main():
    llm_call_using_langchain_chat_models()

def llm_call_using_ollama_sdk():
    response =ollama.chat(model='qwen2.5:3b',messages=[{
    'role': 'user',
    'content': 'Why is the sky blue?',}])
    print(response['message']['content'])

def llm_call_using_openai_sdk():
    load_dotenv() # to make sure all env variables are loaded(especially if .env not exists)
    openai_api_key = get_api_key_from_env()
    # The client looks for an OPENAI_API_KEY environment variable by default(optional to pass it)
    client = OpenAI(apiKey=openai_api_key)
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

def llm_call_using_langchain_chat_models():
    ollama_llm= init_chat_model(model='qwen2.5:3b',model_provider="ollama",temperature=0)
    response= ollama_llm.invoke('Why is the sky blue?') #AIMessage
    print(response.content)

def get_api_key_from_env():
    openai_api_key= os.environ.get('OPENAI_API_KEY')
    if not openai_api_key:
        print("OpenAI API key not loaded")
    else:
        raise ValueError("OpenAI API key not loaded")
    return openai_api_key

def load_env_variables_using_load_dotenv():
    load_dotenv()

if __name__ == '__main__':
    main()