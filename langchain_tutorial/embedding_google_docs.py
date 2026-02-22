from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables from .env file
os.getenv("GOOGLE_API_KEY")

# embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-3-large", dimensions=32)
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", dimensions=32)

document = [
    "Hello, world! This is a test document.",
    "Langchain is a powerful library for building applications with language models.",
    "Embeddings are a way to represent text as vectors in a high-dimensional space.",
    "Google's Gemini models are state-of-the-art for natural language processing tasks.",
]

vector = embeddings.embed_documents(document)

print(len(vector))  # Should print the number of documents
print(len(vector[0]))  # Should print the dimensionality of the embeddings (32 in this case)
print(str(vector[0][:10]) + "...")  # Print the first 10 dimensions of the first document's embedding for a quick check
