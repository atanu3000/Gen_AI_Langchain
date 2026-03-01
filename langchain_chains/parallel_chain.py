from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1-0528",
    task="text-generation",
)

model_hf = ChatHuggingFace(llm=llm)

model = GoogleGenerativeAI(model='gemini-2.5-flash-lite')

prompt1 = PromptTemplate(
    template="Write a simple notes from the following text on {topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Generate 5 short question answer from the following text. \n {topic}",
    input_variables=["topic"]
)

prompt3 = PromptTemplate(
    template="Merge the provided notes and the quiz into a single document. \nnotes -> {notes} \nquiz -> {quiz}",
    input_variables=["notes", "quiz"]
)

parser = StrOutputParser()

parallel_chain = RunnableParallel(
    {
        "notes": prompt1 | model_hf | parser,
        "quiz": prompt2 | model | parser
    }
)

merge_chain = prompt3 | model | parser

final_chain = parallel_chain | merge_chain

response = final_chain.invoke({"topic": "SVM Algorithm in Machine Learning"})
print("Final Response:\n", response)

final_chain.get_graph().print_ascii()