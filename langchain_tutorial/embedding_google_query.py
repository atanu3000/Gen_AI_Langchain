from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables from .env file
os.getenv("GOOGLE_API_KEY")

# embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-3-large", dimensions=32)
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", dimensions=32)
vector = embeddings.embed_query("hello, world!")

print(str(vector[:10]) + "...")