# query_handler.py
#import necessary libraries and modules
import os
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_cohere import ChatCohere
from dotenv import load_dotenv

# load environment variables from a .env file
load_dotenv()
# Set API keys for various LLMs
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ['COHERE_API_KEY'] = os.getenv('COHERE_API_KEY')
groq_api= os.getenv("GROQ_API_KEY")

def select_llm(llm_type):
    """
    Select and initialize the desired LLM based on user selection.
    
    Args:
    - llm_type: The type of LLM to be used.
    
    Returns:
    - An instance of the selected LLM.
    """
    if llm_type == "OpenAI GPT-3.5":
        #initialize openai gpt 3.5
        return ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    # elif llm_type == "Google Gemini":
    #     return ChatGoogleGenerativeAI(model="gemini-pro")
    elif llm_type=="Llama-3-70b":
        #initialize llama3 70b model using groq
        return ChatGroq(temperature=0, groq_api_key=groq_api, model_name='llama3-70b-8192')
    elif llm_type == "Groq Mixtral":
        #initialize mixtral model using groq
        return ChatGroq(temperature=0, groq_api_key=groq_api, model_name="mixtral-8x7b-32768")
    elif llm_type=="Llama-3-8b":
        #initialize llama3 8b model using groq
        return ChatGroq(temperature=0, groq_api_key=groq_api, model_name='llama3-8b-8192')
    else:
        raise ValueError("Unsupported LLM type selected")
