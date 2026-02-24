from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.7)

chat_history = [
    SystemMessage(content="You are a helpful assistant that provides concise and accurate information."),
]

while True:
    user_input = input("You: ")
    chat_history.append(HumanMessage(content=user_input))
    if user_input.lower() in ["exit", "quit"]: break

    response = model.invoke(chat_history)
    print("AI:", response.content)
    chat_history.append(AIMessage(content=response.content))

