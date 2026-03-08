from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence

load_dotenv()

prompt1 = PromptTemplate(
    template="Write a joke about {topic}.",
    input_variables=["topic"],
)

prompt2 = PromptTemplate(
    template="Now, write the joke and a short explanation of the joke: {joke}",
    input_variables=["joke"],
)

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
parser = StrOutputParser()

chain  = RunnableSequence(prompt1, model, parser, prompt2, model, parser)
print(chain.invoke({"topic": "Women in tech"}))
