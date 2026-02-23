from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

st.header("Langchain Prompts UI")
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.7)

prompt = st.text_input("Enter your prompt here:")

# if st.button("Generate Response"):
#     response = model.invoke(prompt)
#     st.write("Response:")
#     st.write(response.text)

if st.button("Generate Response"): # Stream response in real-time
    for chunk in model.stream(prompt):
        st.write(chunk.text)