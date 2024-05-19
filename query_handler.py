# query_handler.py
# here required modules are initialised
import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.load import dumps, loads

load_dotenv()

# these are the separate files required to run this program
from corrective_rag import c_rag_func
from adaptive_rag import adaptive_rag_func
from self_rag import self_rag_new


# here we define the process to direct to various types of rag 
def handle_query(query, rag_type, retriever,llm): 
    """
    Handle the query based on the specified RAG type.
    
    Args:
    - query: The input query to be processed.
    - rag_type: The type of RAG to be used.
    - retriever: The document retriever to use.
    - llm: The large language model to use.
    
    Returns:
    - The result of the specified RAG process.
    """ 
    if rag_type == "Simple RAG":
        # Simple RAG implementation
        return simple_rag(query,retriever,llm)

    elif rag_type == "Multi-Query RAG":
        #Multi-Query RAG implementation
        return multi_query_rag(query,retriever,llm)

    elif rag_type == 'RAG Fusion':
       # RAG Fusion implementation
       return rag_fusion(query,retriever,llm)

    elif rag_type == "Recursive Decomposition RAG":
        #Recursive Decomposition rag implementation
        return decomposition_rag_1(query,retriever,llm)       

    elif rag_type == "Iterative Decomposition RAG":
        #Iterative Decomposition Rag implementation
        return decomposittion_rag_2(query,retriever,llm)

    elif rag_type == "Stepback RAG":
        #Stepback Rag implementation
        return stepback_rag(query,retriever,llm)

    elif rag_type == 'Hypothetical Document Embeddings RAG':
       #Hypothetical Documents Embeddings Rag implementation
       return hyde_rag(query,retriever, llm)
    
    elif rag_type == 'C-RAG':
        #Corrective Rag implementation
        return c_rag_func(query,retriever,llm)
    
    elif rag_type == 'Adaptive RAG':
        #Adaptive Rag implementation
        return adaptive_rag_func(query,retriever,llm)
    
    elif rag_type == "Self RAG":
        #Self Rag implementation
        #  st.write("we reach query handle")
         return self_rag_new(query,retriever,llm)



# It is the simple rag implementation
def simple_rag(query,retriever,llm):
        # required modules are loaded
        from langchain import hub
        from langchain_core.runnables import RunnablePassthrough
        from langchain_core.output_parsers import StrOutputParser
        
        # from the langchain hub the rag  prompt in loaded
        prompt = hub.pull("rlm/rag-prompt")
        
        # convert the whole doc in single string
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        # rag chain is defined
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}  # retrive the document from the list
            | prompt # use the prompt template
            | llm  # use the language model to generate the final answer
            | StrOutputParser()  # Parse the output as string
        )
        return rag_chain.invoke(query)


#Multi-Query RAG implementation
def multi_query_rag(query,retriever,llm):
        from langchain.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        from operator import itemgetter
        # Multi-Query RAG implementation

        # Define the prompt template for generating mulitpole perpective of the user query
        template = """You are an AI language model assistant. Your task is to generate five 
        different versions of the given user question to retrieve relevant documents from a vector 
        database. By generating multiple perspectives on the user question, your goal is to help
        the user overcome some of the limitations of the distance-based similarity search. 
        Provide these alternative questions separated by newlines. Original question: {question}"""

        # Create a prompt instance using the template
        prompt_perspectives = ChatPromptTemplate.from_template(template)
        
        # Define the chain to generate multiple quieries, parse the output, and split into a list of queries
        generate_queries = (
            prompt_perspectives # start with the prompt template
            | llm  # Use language model
            | StrOutputParser() # output as string
            | (lambda x: x.split("\n")) # split the generated string into a list of queries
        )

        def get_unique_union(documents: list[list]):
            """ Unique union of retrieved docs """
            # Flatten list of lists, and convert each Document to string for easy comparison
            flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
            # Get unique documents by converting the list to a set and back to a list
            unique_docs = list(set(flattened_docs))
            # Convert the string representation back to Documents objects and return them
            return [loads(doc) for doc in unique_docs]

        # Define the retrievel chain
        # This chain generates queries, retrieves documents for each query, adn gets a unique set of documents
        retrieval_chain = generate_queries | retriever.map() | get_unique_union

        # Retrieve documents using the retrieval chain
        docs = retrieval_chain.invoke(query)
        
        # Define a prompt template for generating the final answer based on the retrieved documents
        template = """Answer the following question based on this context:

        {context}

        Question: {question}
        """
        # create a prompt instance using the template for the final answer
        prompt = ChatPromptTemplate.from_template(template)

        #llm = ChatOpenAI(temperature=0)
        
        # Define the final RAG (Retrieval-Augmented Generation) chain
        final_rag_chain = (
            {"context": retrieval_chain, 
            "question": itemgetter("question")}  # Extract the question from the input dictionary
            | prompt  # Use the prompt template
            | llm  # Use the language model to generate the final answer
            | StrOutputParser()  # Parse the output as a string
        )
    
        # Invoke the final RAG chain with the original question and return the generated answer
        return final_rag_chain.invoke({"question": query})


# RAG Fusion implementation
def rag_fusion(query,retriever,llm):
    
     
        from langchain.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        from operator import itemgetter
        # RAG-Fusion: Related
        
        # Define the template for generating the mutliple search queries
        template = """You are a helpful assistant that generates multiple search queries based on a single input query. \n
        Generate multiple search queries related to: {question} \n
        Output (4 queries):"""

        # Create a prompt instance using the template
        prompt_rag_fusion = ChatPromptTemplate.from_template(template)

        generate_queries = (
            prompt_rag_fusion  # Start with the prompt template
            | llm  # Use the language model to generate alternative queries
            | StrOutputParser()  # Parse the output as a string
            | (lambda x: x.split("\n"))  # Split the generated string into a list of queries
        )

        def reciprocal_rank_fusion(results: list[list], k=60):
            """ Reciprocal_rank_fusion that takes multiple lists of ranked documents 
                and an optional parameter k used in the RRF formula """
            
            # Initialize a dictionary to hold fused scores for each unique document
            fused_scores = {}

            # Iterate through each list of ranked documents
            for docs in results:
                # Iterate through each document in the list, with its rank (position in the list)
                for rank, doc in enumerate(docs):
                    # Convert the document to a string format to use as a key (assumes documents can be serialized to JSON)
                    doc_str = dumps(doc)
                    # If the document is not yet in the fused_scores dictionary, add it with an initial score of 0
                    if doc_str not in fused_scores:
                        fused_scores[doc_str] = 0
                    # Retrieve the current score of the document, if any
                    previous_score = fused_scores[doc_str]
                    # Update the score of the document using the RRF formula: 1 / (rank + k)
                    fused_scores[doc_str] += 1 / (rank + k)

            # Sort the documents based on their fused scores in descending order to get the final reranked results
            reranked_results = [
                (loads(doc), score)
                for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
            ]

            # Return the reranked results as a list of tuples, each containing the document and its fused score
            return reranked_results

        # Define the retrieval chian using the generate queries chain and reciprocal rank fusion
        retrieval_chain_rag_fusion = generate_queries | retriever.map() | reciprocal_rank_fusion

        # Retrieve documents using the retrieval chain
        docs = retrieval_chain_rag_fusion.invoke({"question": query})

        from langchain_core.runnables import RunnablePassthrough

        # RAG
        # Define a template for generating the final answer based on the retrieved documents
        template = """Answer the following question based on this context:

        {context}

        Question: {question}
        """

        #Create a prompt instance using the template for the final answer
        prompt = ChatPromptTemplate.from_template(template)

        # Define the final RAG (Retrieval-Augmented Generation) chain
        final_rag_chain = (
            {"context": retrieval_chain_rag_fusion, 
            "question": itemgetter("question")}  # Extract the question from the input dictionary
            | prompt  # Use the prompt template
            | llm  # Use the language model to generate the final answer
            | StrOutputParser()  # Parse the output as a string
        )

        # Invoke the final RAG chain with the original question and return the generated answer
        return final_rag_chain.invoke({"question": query})

# Recursive Decomposition implmentation
def decomposition_rag_1(query,retriever, llm):
        from langchain.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        from operator import itemgetter

        # Decomposition template for generating multiple sub-questions from an input question
        template = """You are a helpful assistant that generates multiple sub-questions related to an input question. \n
        The goal is to break down the input into a set of sub-problems / sub-questions that can be answers in isolation. \n
        Generate multiple search queries related to: {question} \n
        Output (3 queries):"""

        # Create a prompt instance using the template
        prompt_decomposition = ChatPromptTemplate.from_template(template)

        from langchain_core.output_parsers import StrOutputParser
        # Define a chain to generate multiple sub-questions, parse the output, and split into a list of sub-questions
        generate_queries_decomposition = (
            prompt_decomposition  # Start with the prompt template
            | llm  # Use the language model to generate sub-questions
            | StrOutputParser()  # Parse the output as a string
            | (lambda x: x.split("\n"))  # Split the generated string into a list of sub-questions
        )
         
        # Run the chain to generate sub-questions from the original query
        question = query
        questions = generate_queries_decomposition.invoke({"question":question})
        
           
        #st.write(questions)

        # Define a template for generating the final answer using the context and sub-questions
        template = """Here is the question you need to answer:

        \n --- \n {question} \n --- \n

        Here is any available background question + answer pairs:

        \n --- \n {q_a_pairs} \n --- \n

        Here is additional context relevant to the question: 

        \n --- \n {context} \n --- \n

        Use the above context and any background question + answer pairs to answer the question: \n {question}
        """

        # Create a prompt  instance using the template for the final answer generationg
        decomposition_prompt = ChatPromptTemplate.from_template(template)

        from operator import itemgetter
        from langchain_core.output_parsers import StrOutputParser

        def format_qa_pair(question, answer):
            """Format Q and A pair"""
            
            formatted_string = ""
            formatted_string += f"Question: {question}\nAnswer: {answer}\n\n"
            return formatted_string.strip()

       
        from langchain_core.output_parsers import StrOutputParser

        # Initialize an empty string to store question and answer pairs
        q_a_pairs = ""
        for q in questions:
            # Define the chain for retrieving context and generating the final answer
            rag_chain = (
                {"context": itemgetter("question") | retriever,  # Retrieve context based on the sub-question
                "question": itemgetter("question"),  # Pass the sub-question to the chain
                "q_a_pairs": itemgetter("q_a_pairs")}  # Include any existing Q&A pairs
                | decomposition_prompt  # Use the prompt template for answer generation
                | llm  # Use the language model to generate the answer
                | StrOutputParser()  # Parse the output as a string
            )

            # Invoke the chain to get the answer for the sub-question
            answer = rag_chain.invoke({"question": q, "q_a_pairs": q_a_pairs})
            # Format the question and answer pair
            q_a_pair = format_qa_pair(q, answer)
            # Append the new Q&A pair to the existing pairs
            q_a_pairs = q_a_pairs + "\n---\n" + q_a_pair

        # Return the final answer
        return answer

# Iterative Decomposition RAG implementation
def decomposittion_rag_2(query,retriever,llm):
     
        # Answer each sub-question individually 
    from langchain import hub
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    # RAG prompt template pulled from a remote hub
    prompt_rag = hub.pull("rlm/rag-prompt")

    def retrieve_and_rag(question, prompt_rag, sub_question_generator_chain):
        """
        Perform RAG (Retrieval-Augmented Generation) on each sub-question.
        Args:
        - question: The main input question.
        - prompt_rag: The RAG prompt template.
        - sub_question_generator_chain: The chain for generating sub-questions.
        
        Returns:
        - rag_results: List of answers for each sub-question.
        - sub_questions: List of generated sub-questions.
        """
        # Generate sub-questions from the main question using the provided chain
        sub_questions = sub_question_generator_chain.invoke({"question": question})
        
        # Initialize a list to hold RAG chain results
        rag_results = []
        
        for sub_question in sub_questions:
            # Retrieve documents relevant to each sub-question
            retrieved_docs = retriever.get_relevant_documents(sub_question)
            
            # Use retrieved documents and sub-question in RAG chain to get the answer
            answer = (prompt_rag | llm | StrOutputParser()).invoke({"context": retrieved_docs, 
                                                                    "question": sub_question})
            # Append the answer to the results list
            rag_results.append(answer)
        
        return rag_results, sub_questions

    # Decomposition template for generating multiple sub-questions from an input question
    template = """You are a helpful assistant that generates multiple sub-questions related to an input question. \n
    The goal is to break down the input into a set of sub-problems / sub-questions that can be answered in isolation. \n
    Generate multiple search queries related to: {question} \n
    Output (3 queries):"""

    # Create a prompt instance using the template
    prompt_decomposition = ChatPromptTemplate.from_template(template)

    # Define a chain to generate multiple sub-questions, parse the output, and split into a list of sub-questions
    generate_queries_decomposition = (
        prompt_decomposition  # Start with the prompt template
        | llm  # Use the language model to generate sub-questions
        | StrOutputParser()  # Parse the output as a string
        | (lambda x: x.split("\n"))  # Split the generated string into a list of sub-questions
    )

    # Wrap the retrieval and RAG process in a RunnableLambda for integration into a chain
    answers, questions = retrieve_and_rag(query, prompt_rag, generate_queries_decomposition)

    def format_qa_pairs(questions, answers):
        """
        Format question and answer pairs.
        Args:
        - questions: List of sub-questions.
        - answers: List of answers corresponding to the sub-questions.
        
        Returns:
        - formatted_string: A single string containing formatted Q&A pairs.
        """
        formatted_string = ""
        for i, (question, answer) in enumerate(zip(questions, answers), start=1):
            formatted_string += f"Question {i}: {question}\nAnswer {i}: {answer}\n\n"
        return formatted_string.strip()

    # Format the Q&A pairs into a single string
    context = format_qa_pairs(questions, answers)

    # Final prompt template for synthesizing an answer from the Q&A pairs and additional context
    template = """Here is a set of Q+A pairs:

    {context}

    Use these to synthesize an answer to the question: {question}
    """

    # Create a prompt instance using the template for the final answer generation
    prompt = ChatPromptTemplate.from_template(template)

    # Define the final RAG (Retrieval-Augmented Generation) chain
    final_rag_chain = (
        prompt  # Use the prompt template
        | llm  # Use the language model to generate the final answer
        | StrOutputParser()  # Parse the output as a string
    )

    # Invoke the final RAG chain with the context and original question, and return the generated answer
    return final_rag_chain.invoke({"context": context, "question": query})

       

#Stepback rag implementation     
def stepback_rag(query,retriever,llm):
        
     
        question = query

        # Few Shot Examples
        from langchain_core.runnables import RunnablePassthrough, RunnableLambda
        from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

        # Examples of how to transform specific questions to more generic ones
        examples = [
            {
                "input": "Could the members of The Police perform lawful arrests?",
                "output": "what can the members of The Police do?",
            },
            {
                "input": "Jan Sindel’s was born in what country?",
                "output": "what is Jan Sindel’s personal history?",
            },
        ]

        # Transform these examples to example messages
        example_prompt = ChatPromptTemplate.from_messages(
            [
                ("human", "{input}"),
                ("ai", "{output}"),
            ]
        )

        # Create a few-shot prompt template using the examples
        few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            examples=examples,
        )

        # Define the main prompt template for generating step-back questions
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an expert at world knowledge. Your task is to step back and paraphrase a question to a more generic step-back question, which is easier to answer. Here are a few examples:""",
                ),
                # Few shot examples
                few_shot_prompt,
                # New question
                ("user", "{question}"),
            ]
        )

        from langchain_core.output_parsers import StrOutputParser

        # Define the chain for generating step-back questions
        generate_queries_step_back = prompt | llm | StrOutputParser()

        # Generate the step-back question using the chain
        step_back_question = generate_queries_step_back.invoke({"question": question})

        # Define the response prompt template
        response_prompt_template = """You are an expert of world knowledge. I am going to ask you a question. Your response should be comprehensive and not contradicted with the following context if they are relevant. Otherwise, ignore them if they are not relevant.

        # {normal_context}
        # {step_back_context}

        # Original Question: {question}
        # Answer:"""

        # Create a prompt instance using the response prompt template
        response_prompt = ChatPromptTemplate.from_template(response_prompt_template)

        # Define the final chain for generating the response
        chain = (
            {
                # Retrieve context using the normal question
                "normal_context": RunnableLambda(lambda x: x["question"]) | retriever,
                # Retrieve context using the step-back question
                "step_back_context": generate_queries_step_back | retriever,
                # Pass on the question
                "question": lambda x: x["question"],
            }
            | response_prompt  # Use the response prompt template
            | llm  # Use the language model to generate the final answer
            | StrOutputParser()  # Parse the output as a string
        )

        # Invoke the final chain with the original question and return the generated answer
        return chain.invoke({"question": question})

def hyde_rag(query,retriever, llm):
        from langchain.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        from langchain_openai import ChatOpenAI

        # Define the HyDE (Hypothetical Document Embedding) template for generating scientific paper passages
        template = """Please write a scientific paper passage to answer the question
        Question: {question}
        Passage:"""
        # Create a prompt instance using the template
        prompt_hyde = ChatPromptTemplate.from_template(template)

        # Define the chain for generating documents for retrieval
        generate_docs_for_retrieval = (
            prompt_hyde  # Use the HyDE prompt template
            | llm  # Use the language model to generate the passage
            | StrOutputParser()  # Parse the output as a string
        )

        # Run the chain to generate hypothetical documents based on the input question
        question = query
        generate_docs_for_retrieval.invoke({"question": question})

        # Define the retrieval chain to retrieve relevant documents using the generated hypothetical documents
        retrieval_chain = generate_docs_for_retrieval | retriever
        # Retrieve documents based on the generated hypothetical documents
        retrieved_docs = retrieval_chain.invoke({"question": question})

        # Define the final prompt template for generating the answer based on the retrieved documents
        template = """Answer the following question based on this context:

        {context}

        Question: {question}
        """
        # Create a prompt instance using the template for the final answer generation
        prompt = ChatPromptTemplate.from_template(template)

        # Define the final RAG (Retrieval-Augmented Generation) chain
        final_rag_chain = (
            prompt  # Use the final prompt template
            | llm  # Use the language model to generate the final answer
            | StrOutputParser()  # Parse the output as a string
        )

        # Invoke the final RAG chain with the retrieved documents and original question, and return the generated answer
        return final_rag_chain.invoke({"context": retrieved_docs, "question": question})
