from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = GoogleGenerativeAI(model='gemini-2.5-flash-lite')
prompt = PromptTemplate(
    template="Write 5 fun facts about {topic}",
    input_variables=["topic"]
)

parser = StrOutputParser()

chain = prompt | model | parser
result = chain.invoke({"topic": "Space"})

print(result)

# chain.get_graph().print_ascii()