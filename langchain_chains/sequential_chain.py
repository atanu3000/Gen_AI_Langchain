from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1-0528",
    task="text-generation",
)

model_hf = ChatHuggingFace(llm=llm)

# 1st prompt
template1 = PromptTemplate(
    input_variables=["input"],
    template="write a detailed report on {input}",
)

# 2nd prompt
template2 = PromptTemplate(
    input_variables=["input"],
    template="write only 5 line brief summery on the text. /n{input}",
)


## Using chain to connect the prompts and models
from langchain_core.output_parsers import StrOutputParser

parser = StrOutputParser()
chain = template1 | model_hf | parser | template2 | model_hf | parser

chain_response = chain.invoke({"input": "lust of human eye"})
print("Chain Response:\n", chain_response)

chain.get_graph().print_ascii()