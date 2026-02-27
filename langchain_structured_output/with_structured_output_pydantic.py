from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional, Literal
from pydantic import BaseModel, Field

load_dotenv()  # Load environment variables from .env file

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")

# pydantic schema for the structured output
class Review(BaseModel):

    key_themes: list[str] = Field(..., description="The main themes of the review, e.g. 'performance', 'battery life', 'camera'")
    summary: str = Field(..., description="A brief summary of the review, no more than 20 words")
    sentiment: Literal["positive", "negative", "neutral"] = Field(..., description="The sentiment of the review.")
    pros: Optional[list[str]] = Field(None, description="A list of pros mentioned in the review")
    cons: Optional[list[str]] = Field(None, description="A list of cons mentioned in the review")

structured_model = model.with_structured_output(Review)


result = structured_model.invoke("""I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it's an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast-whether I'm gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.
The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera-the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.
However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung's One UI still comes with bloatware-why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.
Pros:
Insanely powerful processor (great for gaming and productivity)
Stunning 200MP camera with incredible zoom capabilities
Long battery life with fast charging
S-Pen support is unique and useful
Cons:
Bulky and heavy-not great for one-handed use
Bloatware still exists in One UI
Expensive compared to competitors
I would recommend the Galaxy S24 Ultra to power users and photography enthusiasts who can make the most of its features, but casual users might find it overkill and pricey.""")
print(result)