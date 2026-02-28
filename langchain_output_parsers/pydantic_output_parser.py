from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, EmailStr, Field
from typing import Optional


load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1-0528",
    task="text-generation",
)

model_hf = ChatHuggingFace(llm=llm)

class Student(BaseModel):
    """A model representing a student.
    Attributes:
        name (str): The name of the student.
        age (Optional[int]): The age of the student. Must be a non-negative integer.
        email (Optional[EmailStr]): The email address of the student. Must be a valid email format.
        cgpa (Optional[float]): The CGPA of the student. Must be between 0 and 10.
    """
    name: str = Field("Atanu", description="The name of the student")
    age: Optional[int] = Field(None, ge=3, description="The age of the student. Must be a non-negative integer.")
    email: Optional[EmailStr] = Field(None, description="The email address of the student. Must be a valid email format.")
    cgpa: Optional[float] = Field(None, ge=0, le=10, description="The CGPA of the student. Must be between 0 and 10.")

parser = PydanticOutputParser(pydantic_object=Student)

template = PromptTemplate(
    template="Generate a JSON object representing a student of {place} with the following details: name, age, email and cgpa. {format_instructions}",
    input_variables=["place"],
    partial_variables={'format_instructions': parser.get_format_instructions()}
)

chain = template | model_hf | parser
chain_response = chain.invoke({"place": "India"})
print("Chain Response:\n", chain_response)