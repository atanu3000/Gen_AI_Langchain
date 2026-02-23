from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

# Load environment variables from .env file
load_dotenv()
os.getenv("GOOGLE_API_KEY")

# Initialize the GoogleGenAI Embedding model
embedding = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", dimensions=300)

documents = [
    "Virat kohli is an India cricketer known for his aggressive batting and leadership skills.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills",
    "Sachin Tendulkar is a legendary Indian cricketer often referred to as the 'God of Cricket'.",
    "Rohit Sharma is an Indian cricketer known for his explosive batting and multiple double centuries in ODIs.",
    "Jasprit Bumrah is an Indian fast bowler renowned for his unique bowling action and ability to bowl yorkers consistently."
]

query = "Tell me about bumrah"

doc_embeddings = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)

scores = cosine_similarity([query_embedding], doc_embeddings)[0]
# print("Similarity scores:", scores)

index, score = sorted(list(enumerate(scores)), key=lambda x: x[1])[-1] # Get the index of the most similar document

print("Query:", query)
print("Result:", documents[index])
print("Similarity score:", score.round(4))