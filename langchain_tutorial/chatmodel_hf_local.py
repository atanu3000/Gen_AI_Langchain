from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables from .env file
os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

# os.environ['HF_HOME'] = 'E:/hf_cache'  # Set the cache directory for Hugging Face models

llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs={"temperature": 0.1, "max_new_tokens": 100},
)

model = ChatHuggingFace(llm=llm)

result = model.invoke("Who is the president of the United States?")
print(result.content)