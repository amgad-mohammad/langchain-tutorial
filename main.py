import ollama

def main():
    #---Ollama SDK---
    response =ollama.chat(model='qwen2.5:3b',messages=[{
    'role': 'user',
    'content': 'Why is the sky blue?',}])
    print(response['message']['content'])
    # ---OpenAI SDK---
    


if __name__ == '__main__':
    main()