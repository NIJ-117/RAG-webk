# importing the necessary libraries
import os
import streamlit as st
from langchain_chroma import Chroma
from langchain.load import dumps, loads
from langchain_openai import  OpenAIEmbeddings
from dotenv import load_dotenv

#load the environment variables form a .env file
load_dotenv()

# this are the separate file required for app
from llms import select_llm
from query_handler import handle_query

# Set API keys formt the environment variables
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ['COHERE_API_KEY'] = os.getenv('COHERE_API_KEY')

# using the st.cache_resource so that the database is not loaded everytime
@st.cache_resource
def load_embedding_and_vectorstore():
    # embedding_function = CohereEmbeddings()
    # vectorstore = Chroma(persist_directory="./chdartscohere", embedding_function=embedding_function)
    vectorstore = Chroma(persist_directory="./chdopenarts",embedding_function=OpenAIEmbeddings())
    print("new store is loaded openai embedding is loaded")
    return vectorstore

@st.cache_resource  # this funtion allows the user to select the type of database he would like to use
def load_embedding_and_vectorstore(embedding_type):
    if embedding_type == 'OpenAI':
        from langchain_openai import OpenAIEmbeddings
        vectorstore = Chroma(persist_directory="./chdopenarts",embedding_function=OpenAIEmbeddings())
    elif embedding_type == 'Cohere':
        from langchain_cohere import CohereEmbeddings
        embedding_function = CohereEmbeddings()
        vectorstore = Chroma(persist_directory="./chdartscohere", embedding_function=embedding_function)
    else:
        raise ValueError("Unsupported embedding type")

    print(f"Vector store with {embedding_type} embeddings is loaded")
    return vectorstore

# Allow user to select a RAG type
rag_type = st.selectbox(
    "Select the RAG type",
    ("Simple RAG", "Multi-Query RAG", "RAG Fusion", "Recursive Decomposition RAG", "Iterative Decomposition RAG", "Stepback RAG", "Hypothetical Document Embeddings RAG","Self RAG","C-RAG","Adaptive RAG")
)


# due to in development process every llm code is not customized for every rag type. So, here we my adjustment according to that on particular which option have to be given
if rag_type == "C-RAG":
    embedding_type = st.selectbox("Select the embedding type", ("Cohere",))
    llm_type = st.selectbox("Select the LLM type", ("OpenAI GPT-3.5",))

elif rag_type in ("Self RAG","Adaptive RAG"):
    embedding_type = st.selectbox("Select the embedding type", ("Cohere",))
    llm_type = st.selectbox("Select the LLM type", ("Llama-3-70b","OpenAI GPT-3.5",))

elif rag_type in ("Recursive Decomposition RAG", "Iterative Decomposition RAG"):
    embedding_type = st.selectbox("Select the embedding type", ("OpenAI", "Cohere"))
    llm_type = st.selectbox("Select the LLM type", ("OpenAI GPT-3.5", "Groq Mixtral"))
    
else:
    embedding_type = st.selectbox("Select the embedding type", ("OpenAI", "Cohere"))
    llm_type = st.selectbox("Select the LLM type", ("OpenAI GPT-3.5", "Llama-3-70b", "Llama-3-8b", "Groq Mixtral"))

# laod the vectorstore based on the selected embeddings type
vectorstore = load_embedding_and_vectorstore(embedding_type)
# setting the retriver value here k can be adjusted to decided how many retrieve document the system may provide on activating
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

llm = select_llm(llm_type)  # Initialize the selected LLM


# User input for the query
user_query = st.text_input("Enter your query here:")
if st.button("Get Response"):
    # Process the query based on the selected RAG type

    response = handle_query(user_query, rag_type, retriever, llm)
    # handle_query is a separate file where various type of rag function are iniatialized.

    st.write(response)