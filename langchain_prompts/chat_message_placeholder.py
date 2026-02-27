from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

chat_template = ChatPromptTemplate([
    ("system", "You are a helpful customer support assistant."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{query}"),
])

# Load the chat history 
chat_history = []

with open("chat_history.txt", "r") as f:
    chat_history.extend(f.readlines())

print(chat_history)

prompt = chat_template.invoke({"query": "What is the return policy?", "chat_history": chat_history})

print(prompt)