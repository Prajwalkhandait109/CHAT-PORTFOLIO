from typing import Dict, Any, Optional
from langgraph.graph import StateGraph, END
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from groq import Groq
import logging

from .nodes import (
    GraphState, QueryAnalysisNode, RetrievalNode, GradingNode, 
    GenerationNode, HallucinationCheckNode, RewriteDecisionNode
)
from .query_rewriter import QueryRewriter
from .grade import DocumentGrader

logger = logging.getLogger(__name__)


class AdvancedRAGPipeline:
    """Advanced RAG pipeline with multi-stage processing and quality control."""
    
    def __init__(self, groq_client: Groq, vector_store: Optional[FAISS] = None):
        self.groq_client = groq_client
        self.vector_store = vector_store
        
        # Initialize components
        self.query_rewriter = QueryRewriter(groq_client)
        self.document_grader = DocumentGrader(groq_client)
        
        # Initialize nodes
        self.query_analysis_node = QueryAnalysisNode(self.query_rewriter)
        self.retrieval_node = RetrievalNode(self.vector_store, k=5)
        self.grading_node = GradingNode(self.document_grader, relevance_threshold=0.6)
        self.generation_node = GenerationNode(groq_client)
        self.hallucination_check_node = HallucinationCheckNode(self.document_grader)
        self.rewrite_decision_node = RewriteDecisionNode(max_rewrites=2)
        
        # Build the graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        logger.info("Building advanced RAG graph")
        
        # Create the graph
        workflow = StateGraph(GraphState)
        
        # Add nodes
        workflow.add_node("query_analysis", self.query_analysis_node)
        workflow.add_node("retrieve", self.retrieval_node)
        workflow.add_node("grade_documents", self.grading_node)
        workflow.add_node("generate", self.generation_node)
        workflow.add_node("hallucination_check", self.hallucination_check_node)
        
        # Add edges
        workflow.add_edge("query_analysis", "retrieve")
        workflow.add_edge("retrieve", "grade_documents")
        workflow.add_edge("grade_documents", "generate")
        workflow.add_edge("generate", "hallucination_check")
        
        # Add conditional edge for rewrite decision
        workflow.add_conditional_edges(
            "hallucination_check",
            self.rewrite_decision_node,
            {
                "rewrite": "query_analysis",
                "end": END
            }
        )
        
        # Set entry point
        workflow.set_entry_point("query_analysis")
        
        return workflow.compile()
    
    def invoke(self, question: str, optimization_strategy: str = "hybrid") -> Dict[str, Any]:
        """
        Run the advanced RAG pipeline.
        
        Args:
            question: User's question
            optimization_strategy: Query optimization strategy
            
        Returns:
            Pipeline results with answer and metadata
        """
        logger.info(f"Running advanced RAG pipeline for question: {question[:50]}...")
        
        # Initialize state
        initial_state = GraphState(
            question=question,
            original_question=question,
            optimized_queries=[],
            retrieved_documents=[],
            graded_documents=[],
            relevant_documents=[],
            generation="",
            rewrite_count=0,
            max_rewrites=2,
            optimization_strategy=optimization_strategy,
            metadata={}
        )
        
        try:
            # Run the graph
            final_state = self.graph.invoke(initial_state)
            
            # Prepare results
            results = {
                "question": question,
                "answer": final_state["generation"],
                "relevant_documents": len(final_state["relevant_documents"]),
                "total_retrieved": len(final_state["retrieved_documents"]),
                "rewrite_count": final_state["rewrite_count"],
                "optimization_strategy": optimization_strategy,
                "metadata": final_state["metadata"]
            }
            
            logger.info(f"Pipeline completed. Found {results['relevant_documents']} relevant documents")
            return results
            
        except Exception as e:
            logger.error(f"Error in advanced RAG pipeline: {e}")
            return {
                "question": question,
                "answer": "I encountered an error processing your question. Please try again.",
                "relevant_documents": 0,
                "total_retrieved": 0,
                "rewrite_count": 0,
                "optimization_strategy": optimization_strategy,
                "metadata": {"error": str(e)}
            }
    
    def set_vector_store(self, vector_store: FAISS):
        """Update the vector store."""
        self.vector_store = vector_store
        self.retrieval_node = RetrievalNode(vector_store, k=5)
        # Rebuild graph with new vector store
        self.graph = self._build_graph()
        logger.info("Updated vector store in pipeline")


class SimpleRAGPipeline:
    """Simplified RAG pipeline for basic use cases."""
    
    def __init__(self, groq_client: Groq, vector_store: Optional[FAISS] = None):
        self.groq_client = groq_client
        self.vector_store = vector_store
        self.document_grader = DocumentGrader(groq_client)
    
    def simple_retrieve_and_generate(self, question: str, k: int = 3) -> Dict[str, Any]:
        """
        Simple RAG without advanced optimization.
        
        Args:
            question: User's question
            k: Number of documents to retrieve
            
        Returns:
            Simple RAG results
        """
        if not self.vector_store:
            return {
                "question": question,
                "answer": "Vector store not available.",
                "relevant_documents": 0,
                "metadata": {"error": "No vector store"}
            }
        
        try:
            # Simple retrieval
            documents = self.vector_store.similarity_search(question, k=k)
            
            if not documents:
                return {
                    "question": question,
                    "answer": "I couldn't find relevant information in the portfolio.",
                    "relevant_documents": 0,
                    "metadata": {"retrieval_count": 0}
                }
            
            # Grade documents for relevance
            graded_docs = self.document_grader.filter_relevant_documents(question, documents, relevance_threshold=0.5)
            
            if not graded_docs:
                return {
                    "question": question,
                    "answer": "I found some documents but they don't seem directly relevant to your question.",
                    "relevant_documents": 0,
                    "metadata": {"graded_count": len(documents)}
                }
            
            # Use relevant documents for generation
            relevant_docs = [doc for doc, _ in graded_docs]
            context = "\n\n---\n\n".join([doc.page_content for doc in relevant_docs])
            
            # Generate answer
            system_prompt = """You are Prajwal's AI portfolio assistant. Use the provided context to answer questions about Prajwal's professional background."""
            
            prompt = f"""
Context from Prajwal's portfolio:
{context}

Question: {question}

Please provide a detailed answer based on the context above."""
            
            response = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=400
            )
            
            answer = response.choices[0].message.content.strip()
            
            return {
                "question": question,
                "answer": answer,
                "relevant_documents": len(relevant_docs),
                "total_documents": len(documents),
                "metadata": {
                    "retrieval_count": len(documents),
                    "relevant_count": len(relevant_docs),
                    "grading_threshold": 0.5
                }
            }
            
        except Exception as e:
            logger.error(f"Error in simple RAG: {e}")
            return {
                "question": question,
                "answer": "I encountered an error processing your question. Please try again.",
                "relevant_documents": 0,
                "metadata": {"error": str(e)}
            }