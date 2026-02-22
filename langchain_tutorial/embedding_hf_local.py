from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables from .env file
os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
document = [
    "Hello, world! This is a test document.",
    "Langchain is a powerful library for building applications with language models.",
    "Embeddings are a way to represent text as vectors in a high-dimensional space.",
    "Hugging Face provides a wide range of pre-trained models for various NLP tasks.",
]

vector = embeddings.embed_documents(document)

print(len(vector))  # Should print the number of documents
print(len(vector[0]))  # Should print the dimensionality of the embeddings (384 in this case)
print(str(vector[0][:10]) + "...")  # Print the first 10 dimensions of the first document's embedding for a quick check