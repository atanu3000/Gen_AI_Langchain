from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate, load_prompt

load_dotenv()

st.header("Langchain AI Summarizer")
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.7)


paper_input = st.selectbox("Select Research Paper Name", ["Attention Is All You Need",
"BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"] )

style_input = st. selectbox("Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"])

length_input = st. selectbox("Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"])

# prompt = st.text_input("Enter your prompt here:")
# prompt = f"Explain the research paper '{paper_input}' in a {style_input} style with a {length_input} length."

template = load_prompt("prompt_template.json") # Load the prompt template from the JSON file

# Create a chain using LCEL (LangChain Expression Language)
chain = template | model

if st.button("Generate Response"): # Stream response in real-time
    for chunk in chain.stream({
        'paper_input': paper_input,
        'style_input': style_input,
        'length_input': length_input
    }):
        st.write(chunk.content)