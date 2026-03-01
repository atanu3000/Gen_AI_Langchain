from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

model = GoogleGenerativeAI(model='gemini-2.5-flash-lite')


parser = StrOutputParser()


class Feedback(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(description="Give the sentiment of the feedback, either 'positive' or 'negative'.")

pydantic_parser = PydanticOutputParser(pydantic_object=Feedback)


prompt1 = PromptTemplate(
    template="Give a sentiment analysis only in one word(e.g. positive or negative) for the following feedback: {text} \n {format_instructions}",
    input_variables=["text"],
    partial_variables={"format_instructions": pydantic_parser.get_format_instructions()}
)


classifier_chain = prompt1 | model | pydantic_parser
# chain_response = classifier_chain.invoke({"text": "The product is really good and I am satisfied with the quality."})
# print("Classification Response:\n", chain_response)

propmt2 = PromptTemplate(
    template="Give a response in a single sentence for this positive following feedback: {feedback}",
    input_variables=["feedback"]
)

prompt3 = PromptTemplate(
    template="Give a response in a single sentence for this negative following feedback: {feedback}",
    input_variables=["feedback"]
)

branch_chain = RunnableBranch(
    (lambda x: x.sentiment == "positive", propmt2 | model | parser),
    (lambda x: x.sentiment == "negative", prompt3 | model | parser),
    RunnableLambda(lambda x: "Could not determine the sentiment of the feedback.")
)

chain = classifier_chain | branch_chain
chain_response = chain.invoke({"text": "The product is really good and I am satisfied with the quality."})
print("\nChain Response: ", chain_response)

chain_response = chain.invoke({"text": "The product is really bad and I am not satisfied with the quality."})
print("\nChain Response: ", chain_response)

chain_response = chain.invoke({"text": "The product is okay and I am not sure about the quality."})
print("\nChain Response: ", chain_response)

chain.get_graph().print_ascii()