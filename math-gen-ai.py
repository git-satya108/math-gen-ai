import os
import openai
from openai import OpenAI
import streamlit as st
from dotenv import load_dotenv, find_dotenv
import pandas as pd
import PyPDF2
from docx import Document
from io import StringIO
import math  # Include the math library
import langchain
import langchain_openai
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from sqlalchemy import create_engine
from werkzeug.utils import secure_filename

# Load OpenAI API key
load_dotenv(find_dotenv(), override=True)
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize OpenAI API client
client = openai.Client()

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display the banner image
st.image("imagebanner2.png", use_column_width=True)

# Function to load and read multiple files (PDF, DOCX, TXT, XLSX)
def load_files(uploaded_files):
    all_texts = []
    for uploaded_file in uploaded_files:
        if uploaded_file.type == "application/pdf":
            reader = PyPDF2.PdfFileReader(uploaded_file)
            text = ""
            for page_num in range(reader.numPages):
                page = reader.getPage(page_num)
                text += page.extract_text()
            all_texts.append(text)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = Document(uploaded_file)
            text = "\n".join([para.text for para in doc.paragraphs])
            all_texts.append(text)
        elif uploaded_file.type == "text/plain":
            text = StringIO(uploaded_file.getvalue().decode("utf-8")).read()
            all_texts.append(text)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
            xls = pd.ExcelFile(uploaded_file)
            for sheet_name in xls.sheet_names:
                sheet = pd.read_excel(xls, sheet_name)
                text = sheet.to_string()
                all_texts.append(text)
    return all_texts

# Function to break content into chunks
def break_into_chunks(text, chunk_size=1000, overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = text_splitter.split_text(text)
    return chunks

# Initialize FAISS index with embeddings
def initialize_faiss_index(chunks):
    embeddings = OpenAIEmbeddings()
    texts = [chunk for chunk in chunks]
    vectorstore = FAISS.from_texts(texts, embeddings)
    return vectorstore

# Chat with the assistant using OpenAI API
def chat_with_assistant(prompt, system_message):
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error: {e}")
        return None

# Streamlit app logic
st.title("Math Magic: Your Math Problem Solver")
uploaded_files = st.file_uploader("Upload files", accept_multiple_files=True, type=['pdf', 'docx', 'txt', 'xlsx'])

if uploaded_files:
    texts = load_files(uploaded_files)
    all_chunks = []
    for text in texts:
        chunks = break_into_chunks(text)
        all_chunks.extend(chunks)
    vectorstore = initialize_faiss_index(all_chunks)
    st.write("Files processed and vector store created.")

# Function to solve math problems using GPT-4
def solve_math_problem(prompt):
    system_message = "You are a math tutor who can solve complex math problems by explaining concepts and providing good explanations for students from classes 7 to 12. Include an explanation of the concept behind solving the problem."
    return chat_with_assistant(prompt, system_message)

# Prompt for math problem
prompt = st.text_area("Enter your math problem here:")
if st.button("Solve"):
    if prompt:
        solution = solve_math_problem(prompt)
        if solution:
            st.write(f"Tutor: {solution}")
            # Add to chat history
            st.session_state.chat_history.append({"user": prompt, "tutor": solution})
            # Display chat history
            for entry in st.session_state.chat_history:
                st.write(f"Student: {entry['user']}")
                st.write(f"Tutor: {entry['tutor']}")
                st.write("-" * 30)
