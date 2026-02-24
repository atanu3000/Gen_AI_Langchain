from langchain_core.prompts import ChatPromptTemplate

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful {domain} expert."),
    ("human", "Explain is simple terms, what is {input}?"),
])

prompt = chat_prompt.invoke({"domain": "machine learning", "input": "overfitting"})

print(prompt)