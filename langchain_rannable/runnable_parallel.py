from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence, RunnableParallel
import re

load_dotenv()

model1 = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1-0528",
    task="text-generation",
)

model2 = ChatHuggingFace(llm=llm)

prompt1 = PromptTemplate(
    template="Write a tweet for the following topic: {topic}",
    input_variables=["topic"],
)

prompt2 = PromptTemplate(
    template="Write a LinkedIn post for the following topic: {topic}",
    input_variables=["topic"],
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    "tweet": RunnableSequence(prompt1, model1, parser),
    "linkedin": RunnableSequence(prompt2, model2, parser)
})

res = parallel_chain.invoke({"topic": "Human"})

# Clean the LinkedIn output by removing <think> tags and their content
linkedin_clean = re.sub(r'<think>.*?</think>', '', res["linkedin"], flags=re.DOTALL).strip()

print("Tweet: ", res["tweet"])
print("-------------------------------")
print("\nLinkedIn: ", linkedin_clean)