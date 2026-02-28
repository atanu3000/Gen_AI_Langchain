from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1-0528",
    task="text-generation",
)

model_hf = ChatHuggingFace(llm=llm)
model_google = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")

# 1st prompt
template1 = PromptTemplate(
    input_variables=["input"],
    template="write a detailed report on {input}",
)

# 2nd prompt
template2 = PromptTemplate(
    input_variables=["input"],
    template="write a 5 line summery on the text. /n {input}",
)

## Using HuggingFace for the prompts
# prompt1 = template1.invoke({"input": "the impact of climate change on agriculture"})
# response1 = model_hf.invoke(prompt1)

# prompt2 = template2.invoke({"input": response1.content})
# response2 = model_hf.invoke(prompt2)

# print("Detailed Report:\n", response1.content)
# print("\nSummary:\n", response2.content)


## Using Google Gemini for the same prompts
# prompt1 = template1.invoke({"input": "the impact of AI on Job Market"})
# response1 = model_google.invoke(prompt1)

# prompt2 = template2.invoke({"input": response1.content})
# response2 = model_google.invoke(prompt2)

# print("Detailed Report:\n", response1.content)
# print("\nSummary:\n", response2.content)


## Using chain to connect the prompts and models
from langchain_core.output_parsers import StrOutputParser

parser = StrOutputParser()
# chain = template1 | model_hf | parser | template2 | model_hf | parser
chain = template1 | model_google | parser | template2 | model_google | parser

chain_response = chain.invoke({"input": "the impact of AI on Job Market"})
print("Chain Response:\n", chain_response)