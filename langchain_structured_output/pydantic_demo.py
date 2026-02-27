from pydantic import BaseModel, Field, EmailStr
from typing import Optional

class Student(BaseModel):

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