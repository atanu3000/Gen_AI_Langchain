from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1-0528",
    task="text-generation",
)

model_hf = ChatHuggingFace(llm=llm)
parser = JsonOutputParser()

template = PromptTemplate(
    template="Give me the name, age and city of a fictional person. \n{format_instructions}",
    input_variables=[],
    partial_variables={'format_instructions': parser.get_format_instructions()}
)

prompt = template.format()

result = model_hf.invoke(prompt)
final_result = parser.parse(result.content)

print("Final Result:\n", final_result)
print(type(final_result))

## using chain to connect the prompt, model and parser
chain = template | model_hf | parser
chain_response = chain.invoke({})
print("Chain Response:\n", chain_response)