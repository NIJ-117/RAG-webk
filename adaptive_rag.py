import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.load import dumps, loads

#load environment variables from a .env file
load_dotenv()

transform_count = 0
def adaptive_rag_func(query,retriever,llm):
        """
        Adaptive RAG function to handle different types of queries by routing them
        to the appropriate data source and generating responses using an LLM.
        
        Args:
        - query: The user query to be processed.
        - retriever: The retriever for fetching relevant documents.
        - llm: The large language model used for generating responses.
        """

        import os
        os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
        os.environ['COHERE_API_KEY'] = os.getenv('COHERE_API_KEY')
        os.environ['TAVILY_API_KEY'] = os.getenv('TAVILY_API_KEY')

        

        ### Router

        from typing import Literal

        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.pydantic_v1 import BaseModel, Field
        from langchain_openai import ChatOpenAI


        # Data model
        class RouteQuery(BaseModel):
            """Route a user query to the most relevant datasource."""

            datasource: Literal["vectorstore", "web_search"] = Field(
                ...,
                description="Given a user question choose to route it to web search or a vectorstore .",
            )


        
        #initialize structured llm router
        structured_llm_router = llm.with_structured_output(RouteQuery)

        # Define routing promt
        system = """You are an expert at routing a user question to a vectorstore or web search.
        The vectorstore contains documents related to Indian Architecture, Sculpture, and Pottery, Indian Paintings,Indian Handicrafts , UNESCO's List of World Heritage Sites in India , Indian Music, Indian Dance Forms
        Indian Theatre, Indian Puppetry, Indian Circus, Martial Arts in India, UNESCO's List of Intangible Cultural Heritage
        Languages in India, Religion in India, Buddhism and Jainism, Indian Literature, Schools of Philosophy, Indian Cinema
        Science and Technology through the Ages, Calendars in India, Fairs and Festivals of India, Awards and Honours, Law and Culture
        Cultural Institutions in India, Coins in Ancient and Medieval India, Indian Culture Abroad, India through the eyes of Foreign Travellers

        Use the vectorstore for questions on these topics. Otherwise, use web-search."""
        route_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "{question}"),
            ]
        )

        question_router = route_prompt | structured_llm_router

        ### Retrieval Grader

        # Data model
        class GradeDocuments(BaseModel):
            """Binary score for relevance check on retrieved documents."""

            binary_score: str = Field(
                description="Documents are relevant to the question, 'yes' or 'no'"
            )


        # LLM with function call

        structured_llm_grader = llm.with_structured_output(GradeDocuments)

        # Prompt
        system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
            It is a simple test. \n
            If the document contains any keyword(s) related to the user question, grade it as relevant. \n
            Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
        grade_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
            ]
        )

        retrieval_grader = grade_prompt | structured_llm_grader

        
        ### Generate

        from langchain import hub
        from langchain_core.output_parsers import StrOutputParser

        # Prompt
        prompt = hub.pull("rlm/rag-prompt")



        # Post-processing funtion to format the documents
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)


        # Chain
        rag_chain = prompt | llm | StrOutputParser()



        
        ### Hallucination Grader


        # Data model for grading the halucination in generated answers
        class GradeHallucinations(BaseModel):
            """Binary score for hallucination present in generation answer."""

            binary_score: str = Field(
                description="Answer is grounded in the facts, 'yes' or 'no'"
            )


        # initialize structured LLm grader for hallucination

        structured_llm_grader = llm.with_structured_output(GradeHallucinations)

        # Define the hallucinations grading prompt
        system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
            Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""
        hallucination_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
            ]
        )

        # Define the halucination chain
        hallucination_grader = hallucination_prompt | structured_llm_grader


        ### Answer Grader


        # Data model for grading the quality of answers
        class GradeAnswer(BaseModel):
            """Binary score to assess answer addresses question."""

            binary_score: str = Field(
                description="Answer addresses the question, 'yes' or 'no'"
            )


        # initialize structured llm grader for answers

        structured_llm_grader = llm.with_structured_output(GradeAnswer)

        # Define answer grading prompt
        system = """You are a grader assessing whether an answer addresses / resolves a question \n 
            Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""
        answer_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
            ]
        )

        # Define the answer grading chain
        answer_grader = answer_prompt | structured_llm_grader



        ### Question Re-writer



        # Define question re-writer prompt
        system = """You a question re-writer that converts an input question to a better version that is optimized \n 
            for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."""
        re_write_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                (
                    "human",
                    "Here is the initial question: \n\n {question} \n Formulate an improved question.",
                ),
            ]
        )

        # question rewriter chain
        question_rewriter = re_write_prompt | llm | StrOutputParser()


        # ## Web Search Tool

        ### Search

        from langchain_community.tools.tavily_search import TavilySearchResults

        web_search_tool = TavilySearchResults(k=3)




        from typing_extensions import TypedDict
        from typing import List


        class GraphState(TypedDict):
            """
            Represents the state of our graph.

            Attributes:
                question: question
                generation: LLM generation
                documents: list of documents
            """

            question: str
            generation: str
            documents: List[str]

        
        MAX_TRANSFORMS = 3
        # ## Graph Flow 


        from langchain.schema import Document


        def retrieve(state):
            """
            Retrieve documents

            Args:
                state (dict): The current graph state

            Returns:
                state (dict): New key added to state, documents, that contains retrieved documents
            """
            print("---RETRIEVE---")
            question = state["question"]

            # Retrieval
            documents = retriever.invoke(question)
            return {"documents": documents, "question": question}


        def generate(state):
            """
            Generate answer

            Args:
                state (dict): The current graph state

            Returns:
                state (dict): New key added to state, generation, that contains LLM generation
            """
            print("---GENERATE---")
            question = state["question"]
            documents = state["documents"]

            # RAG generation
            generation = rag_chain.invoke({"context": documents, "question": question})
            return {"documents": documents, "question": question, "generation": generation}


        def grade_documents(state):
            """
            Determines whether the retrieved documents are relevant to the question.

            Args:
                state (dict): The current graph state

            Returns:
                state (dict): Updates documents key with only filtered relevant documents
            """

            print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
            question = state["question"]
            documents = state["documents"]

            # Score each doc
            filtered_docs = []
            for d in documents:
                score = retrieval_grader.invoke(
                    {"question": question, "document": d.page_content}
                )
                grade = score.binary_score
                if grade == "yes":
                    print("---GRADE: DOCUMENT RELEVANT---")
                    filtered_docs.append(d)
                else:
                    print("---GRADE: DOCUMENT NOT RELEVANT---")
                    continue
            return {"documents": filtered_docs, "question": question}


        def transform_query(state):
            """
            Transform the query to produce a better question.

            Args:
                state (dict): The current graph state

            Returns:
                state (dict): Updates question key with a re-phrased question
            """

            print("---TRANSFORM QUERY---")
            question = state["question"]
            documents = state["documents"]

            # Re-write question
            better_question = question_rewriter.invoke({"question": question})

            global transform_count
            transform_count +=1

            return {"documents": documents, "question": better_question}

        def check_transform_limit(state):
            global transform_count
            """
            Check if the transform query has been called three times.

            Args:
                state (dict): The current graph state.

            Returns:
                str: Next node to call ('retrieve' or 'web_search').
            """
            if transform_count < MAX_TRANSFORMS:
                print(f"---TRANSFORM QUERY COUNT: {transform_count}--- CONTINUE TO RETRIEVE")
                print(state["question"])
                return "retrieve"
            else:
                print(f"---TRANSFORM QUERY COUNT: {transform_count}--- SWITCH TO WEB SEARCH")
                return "web_search"
            
        def web_search(state):
            """
            Web search based on the re-phrased question.

            Args:
                state (dict): The current graph state

            Returns:
                state (dict): Updates documents key with appended web results
            """

            print("---WEB SEARCH---")
            question = state["question"]

            # Web search
            global transform_count
            transform_count = 0
            # print(transform_count)
            docs = web_search_tool.invoke({"query": question})
            web_results = "\n".join([d["content"] for d in docs])
            web_results = Document(page_content=web_results)

            return {"documents": web_results, "question": question}


        ### Edges ###


        def route_question(state):
            """
            Route question to web search or RAG.

            Args:
                state (dict): The current graph state

            Returns:
                str: Next node to call
            """

            print("---ROUTE QUESTION---")
            question = state["question"]
            source = question_router.invoke({"question": question})
            if source.datasource == "web_search":
                print("---ROUTE QUESTION TO WEB SEARCH---")
                return "web_search"
            elif source.datasource == "vectorstore":
                print("---ROUTE QUESTION TO RAG---")
                return "vectorstore"


        def decide_to_generate(state):
            """
            Determines whether to generate an answer, or re-generate a question.

            Args:
                state (dict): The current graph state

            Returns:
                str: Binary decision for next node to call
            """

            print("---ASSESS GRADED DOCUMENTS---")
            question = state["question"]
            filtered_documents = state["documents"]

            if not filtered_documents:
                # All documents have been filtered check_relevance
                # We will re-generate a new query
                print(
                    "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---"
                )
                return "transform_query"
            else:
                # We have relevant documents, so generate answer
                print("---DECISION: GENERATE---")
                return "generate"


        def grade_generation_v_documents_and_question(state):
            """
            Determines whether the generation is grounded in the document and answers question.

            Args:
                state (dict): The current graph state

            Returns:
                str: Decision for next node to call
            """

            print("---CHECK HALLUCINATIONS---")
            question = state["question"]
            documents = state["documents"]
            generation = state["generation"]

            score = hallucination_grader.invoke(
                {"documents": documents, "generation": generation}
            )
            grade = score.binary_score

            # Check hallucination
            if grade == "yes":
                print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
                # Check question-answering
                print("---GRADE GENERATION vs QUESTION---")
                score = answer_grader.invoke({"question": question, "generation": generation})
                grade = score.binary_score
                if grade == "yes":
                    print("---DECISION: GENERATION ADDRESSES QUESTION---")
                    return "useful"
                else:
                    print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
                    return "not useful"
            else:
                pprint("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
                return "not supported"


        # ## Build Graph


        from langgraph.graph import END, StateGraph

        workflow = StateGraph(GraphState)

        # Define the nodes
        workflow.add_node("web_search", web_search)  # web search
        workflow.add_node("retrieve", retrieve)  # retrieve
        workflow.add_node("grade_documents", grade_documents)  # grade documents
        workflow.add_node("generate", generate)  # generatae
        workflow.add_node("transform_query", transform_query)  # transform_query

        # Build graph
        workflow.set_conditional_entry_point(
            route_question,
            {
                "web_search": "web_search",
                "vectorstore": "retrieve",
            },
        )
        workflow.add_edge("web_search", "generate")
        workflow.add_edge("retrieve", "grade_documents")
        workflow.add_conditional_edges(
            "grade_documents",
            decide_to_generate,
            {
                "transform_query": "transform_query",
                "generate": "generate",
            },
        )
        workflow.add_conditional_edges(
            "transform_query",
            check_transform_limit,
            {
                "retrieve": "retrieve",
                "web_search": "web_search",
            },
        )
        workflow.add_conditional_edges(
            "generate",
            grade_generation_v_documents_and_question,
            {
                "not supported": "generate",
                "useful": END,
                "not useful": "transform_query",
            },
        )

        # Compile
        app = workflow.compile()


       

        from pprint import pprint

        # Define the input
        
        max_iterations_nodes = 15  # Maximum number of iterations
        node_count = 0  # Counter for iterations
        interation_count=0 
        if query:
            inputs = {"question": query}
        # Run the workflow
        for output in app.stream(inputs):
            
            
            for key, value in output.items():
                # Node
                node_count += 1
                if node_count > max_iterations_nodes:
                    return st.write(f" Due to compute restrictions we are not able to answer to your query Currently not able to answer.")
                st.write(f"Node -> {node_count} '{key}':")
                # Optional: print full state at each node
                
                
        #         pprint(value["keys"], indent=2, width=80, depth=None)

            if node_count > max_iterations_nodes:
                return st.write(f" Due to compute restrictions we are not able to answer to your query Currently not able to answer.")
               

            pprint("\n---\n")

        # Final generation (only if it reaches this point without breaking)
        if node_count < max_iterations_nodes:
           return st.write(value["generation"])
       
        # # Streamlit app interface

               
        
        #     for output in app.stream(inputs):
        #         for key, value in output.items():
        #             st.write(f"Node '{key}':")
                                
        #         st.write("\n---\n")
        #     st.write("Final Generation:")
        #     return (value["generation"])
        # else:
        #     st.write("Please enter a query.")