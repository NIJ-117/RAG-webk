import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.load import dumps, loads

load_dotenv()

def c_rag_func(query,retriever,llm):
        
        os.environ['TAVILY_API_KEY'] = os.getenv('TAVILY_API_KEY')

        ### Retrieval Grader
        
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.pydantic_v1 import BaseModel, Field

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
            If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
            Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
        grade_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
            ]
        )

        retrieval_grader = grade_prompt | structured_llm_grader
        question = "nature"
        docs = retriever.get_relevant_documents(question)
        #doc_txt = docs.page_content



        ### Generate

        from langchain import hub
        from langchain_core.output_parsers import StrOutputParser

        # Prompt
        prompt = hub.pull("rlm/rag-prompt")

        # LLM


        # Post-processing
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)


        # Chain
        rag_chain = prompt | llm | StrOutputParser()

        # Run
        generation = rag_chain.invoke({"context": docs, "question": question})
        print(generation)


        ### Question Re-writer



        # Prompt
        system = """You a question re-writer that converts an input question to a better version that is optimized \n 
            for web search. Look at the input and try to reason about the underlying semantic intent / meaning."""
        re_write_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                (
                    "human",
                    "Here is the initial question: \n\n {question} \n Formulate an improved question.",
                ),
            ]
        )

        question_rewriter = re_write_prompt | llm | StrOutputParser()
        question_rewriter.invoke({"question": question})


        # Web Search Tool


        ### Search

        from langchain_community.tools.tavily_search import TavilySearchResults

        web_search_tool = TavilySearchResults(k=3)


        # # Graph 
        # 
        # Capture the flow in as a graph.
        # 
        # ## Graph state

        from typing_extensions import TypedDict
        from typing import List


        class GraphState(TypedDict):
            """
            Represents the state of our graph.

            Attributes:
                question: question
                generation: LLM generation
                web_search: whether to add search
                documents: list of documents
            """

            question: str
            generation: str
            web_search: str
            documents: List[str]


       
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
            documents = retriever.get_relevant_documents(question)
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
            web_search = "No"
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
                    web_search = "Yes"
                    continue
            return {"documents": filtered_docs, "question": question, "web_search": web_search}


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
            return {"documents": documents, "question": better_question}


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
            documents = state["documents"]

            # Web search
            docs = web_search_tool.invoke({"query": question})
            web_results = "\n".join([d["content"] for d in docs])
            web_results = Document(page_content=web_results)
            documents.append(web_results)

            return {"documents": documents, "question": question}


        ### Edges


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
            web_search = state["web_search"]
            filtered_documents = state["documents"]

            if web_search == "Yes":
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


        # ## Build Graph
        
        from langgraph.graph import END, StateGraph

        workflow = StateGraph(GraphState)

        # Define the nodes
        workflow.add_node("retrieve", retrieve)  # retrieve
        workflow.add_node("grade_documents", grade_documents)  # grade documents
        workflow.add_node("generate", generate)  # generatae
        workflow.add_node("transform_query", transform_query)  # transform_query
        workflow.add_node("web_search_node", web_search)  # web search

        # Build graph
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "grade_documents")
        workflow.add_conditional_edges(
            "grade_documents",
            decide_to_generate,
            {
                "transform_query": "transform_query",
                "generate": "generate",
            },
        )
        workflow.add_edge("transform_query", "web_search_node")
        workflow.add_edge("web_search_node", "generate")
        workflow.add_edge("generate", END)

        # Compile
        app = workflow.compile()

        # Streamlit app interface

               
        if query:
            inputs = {"question": query}
            for output in app.stream(inputs):
                for key, value in output.items():
                    st.write(f"Node '{key}':")
                                
                st.write("\n---\n")
            st.write("Final Generation:")
            return (value["generation"])
        else:
            st.write("Please enter a query.")