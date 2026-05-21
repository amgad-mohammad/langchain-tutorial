from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_ollama import ChatOllama


def using_prompt_template_without_system_prompt():
    """
    USE WHEN: Simple one-off questions without system instructions.
    BEST FOR: Quick queries, prototyping, basic text completion.
    """
    user_input = input("Enter the framework name:")
    dynamic_user_prompt = PromptTemplate.from_template('Who is the creator of {framework}?')
    final_user_prompt = dynamic_user_prompt.invoke({"framework": user_input})  # StringPromptValue
    ollama_llm = ChatOllama(model='qwen2.5:3b', temperature=0)
    llm_response = ollama_llm.invoke(final_user_prompt)
    print(llm_response.content)


def using_prompt_template_with_system_prompt():
    """
    USE WHEN: You need to control model behavior/role/tone (RECOMMENDED - 99% of cases).
    BEST FOR: Chatbots, assistants, production apps, setting personality.
    WHY: Clean syntax, modern convention, easy to maintain.
    """
    user_input = input("Enter the framework name:")
    # this is finally converted to System & Human Messages
    prompt_template = ChatPromptTemplate.from_messages(
        [("system", 'You are a helpful assistant for answering technology questions'),
         ("user", 'Who is the creator of {framework}?')
         ]
    )
    final_prompt_template = prompt_template.invoke({"framework": user_input})
    ollama_llm = ChatOllama(model='qwen2.5:3b', temperature=0)
    llm_response = ollama_llm.invoke(final_prompt_template)
    print(llm_response.content)


def using_explicit_message_templates():
    """
    USE WHEN: Advanced features needed - partial variables, validators, template reuse (RARE - 1% of cases).
    BEST FOR: Complex template config, building frameworks, legacy code.
    WHY AVOID: Verbose, harder to maintain, tuple syntax does this automatically.
    """
    user_input = input("Enter the framework name:")

    # Explicitly create SystemMessagePromptTemplate
    system_template = SystemMessagePromptTemplate(
        prompt=PromptTemplate(
            input_variables=[],
            template='You are a helpful assistant for answering technology questions'
        )
    )

    # Explicitly create HumanMessagePromptTemplate
    human_template = HumanMessagePromptTemplate(
        prompt=PromptTemplate(
            input_variables=["framework"],
            template='Who is the creator of {framework}?'
        )
    )

    # Combine them into ChatPromptTemplate
    prompt_template = ChatPromptTemplate.from_messages([
        system_template,
        human_template
    ])

    final_prompt_template = prompt_template.invoke({"framework": user_input})
    ollama_llm = ChatOllama(model='qwen2.5:3b', temperature=0)
    llm_response = ollama_llm.invoke(final_prompt_template)
    print(llm_response.content)


if __name__ == '__main__':
    # Simple query? → using_prompt_template_without_system_prompt()
    # Need system prompt? → using_prompt_template_with_system_prompt() ✅ (USE THIS)
    # Advanced config? → using_explicit_message_templates() (RARE)

    using_explicit_message_templates()