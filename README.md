# RAG APP Chatbot 

 

## Description 

The RAG APP Chatbot is an application that implements the Retrieval-Augmented Generation (RAG) architecture within a single file. This project demonstrates the capabilities and context of RAG models and their performance across various configurations.  

 

A pre-loaded PDF file, "Indian Art and Culture," is embedded within the system, allowing users to select from different types of embeddings to enhance the chatbot's functionality. 

 

**PDF Name**: Indian Art and Culture  

**PDF Link**: [Indian Art and Culture](https://drive.google.com/file/d/1inDvijg906FjxU3oyDoThe5EPXY9BYEW/view?usp=sharing) 
*** Documentation  Report *** : [https://drive.google.com/file/d/1-jNsu5NFowYIb8WnKf1Vfjjhw9g1Xn-5/view?usp=sharing](https://drive.google.com/file/d/1-jNsu5NFowYIb8WnKf1Vfjjhw9g1Xn-5/view?usp=sharing)
 

**Caution**: This project is currently under development. As such, the chatbot's output may occasionally be undesired. We apologize for any inconvenience. 

 

## Installation 

 

### Step 1: Create Python Virtual Environment 

For Linux systems: 

```bash 

python3 -m venv myenv 

``` 

 

### Step 2: Activate Python Virtual Environment 

```bash 

source myenv/bin/activate 

``` 

 

### Step 3: Install Requirements 

```bash 

pip install -r requirements.txt 

``` 

 

### Step 4: Create a .env File 

Provide all the necessary API keys to use the chatbot. The format of the `.env` file should be: 

``` 

OPENAI_API_KEY= 

GROQ_API_KEY= 

COHERE_API_KEY= 

TAVILY_API_KEY= 

``` 

 

## Usage 

To run the chatbot, use the following command: 

```bash 

streamlit run app.py 

``` 

 

## Features 

The RAG APP Chatbot supports various RAG models and configurations, including: 

 

- Simple RAG 

- Multiquery RAG 

- RAG Fusion 

- Recursive Decomposition RAG 

- Iterative Decomposition RAG 

- Step-back RAG 

- Self-RAG 

- Corrective RAG 

- Adaptive RAG 

 

### Available Embeddings 

- OpenAI 

- Cohere 

 

### Available LLM Models 

- OpenAI 

- Llama3 70B 

- Llama3 8B 

- Mixtral 8 

 

## Contributing 

Contributions are welcome! Please adhere to the following guidelines: 

 

1. Fork the repository. 

2. Create a new branch (`git checkout -b feature-branch`). 

3. Commit your changes (`git commit -m 'Add some feature'`). 

4. Push to the branch (`git push origin feature-branch`). 

5. Open a pull request. 

 

 

## Contact 

For any questions or support, please contact us at [uniyarakapil@gmail.com](uniyarakapil@gmail.com). 

 
