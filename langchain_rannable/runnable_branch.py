from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda, RunnableBranch

load_dotenv()

prompt1 = PromptTemplate(
    template="Write a report about {topic} with detailed information.",
    input_variables=["topic"],
)

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
parser = StrOutputParser()

report_generation_chain  = prompt1 | model | parser

def word_count(text):
    return len(text.split())

parallel_chain = RunnableParallel({
    'report': RunnablePassthrough(),
    'word_count': RunnableLambda(word_count)
})

prompt2 = PromptTemplate(
    template="Write a summary of the report: {report}",
    input_variables=["report"],
)

branch_chain = RunnableBranch(
    (lambda x: len(x.split()) > 300, prompt2 | model | parser | parallel_chain),
    parallel_chain
)

final_chain = report_generation_chain | branch_chain
response = final_chain.invoke({"topic": 'Iran and Iraq war'})
formatted_response = f"Report: {response['report']}\nWord Count: {response['word_count']}"
print(formatted_response)
