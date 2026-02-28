from pydantic import BaseModel, Field, EmailStr
from typing import Optional

class Student(BaseModel):
    """A model representing a student.
    Attributes:
        name (str): The name of the student.
        age (Optional[int]): The age of the student. Must be a non-negative integer.
        email (Optional[EmailStr]): The email address of the student. Must be a valid email format.
        cgpa (Optional[float]): The CGPA of the student. Must be between 0 and 10.
    """
    name: str = 'Atanu'
    age: Optional[int] = None
    email: Optional[EmailStr] = None
    cgpa: Optional[float] = Field(None, ge=0, le=10)

new_student = Student(name="Alice", age=20)
st2 = Student(email="atanu@example.com", cgpa=8.5)


print(new_student)
print(st2)

print(new_student.model_dump_json(), type(new_student.model_dump_json()))
print(st2.model_dump(), type(st2.model_dump()))