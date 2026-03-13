from typing import Dict, Any, List, Optional, TypedDict
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import logging

from .query_rewriter import QueryRewriter
from .grade import DocumentGrader

logger = logging.getLogger(__name__)


class GraphState(TypedDict):
    """State for the multi-stage RAG graph."""
    question: str
    original_question: str
    optimized_queries: List[str]
    retrieved_documents: List[Document]
    graded_documents: List[tuple[Document, Dict[str, Any]]]
    relevant_documents: List[Document]
    generation: str
    rewrite_count: int
    max_rewrites: int
    optimization_strategy: str
    metadata: Dict[str, Any]


class QueryAnalysisNode:
    """Node for analyzing and optimizing queries."""
    
    def __init__(self, query_rewriter: QueryRewriter):
        self.query_rewriter = query_rewriter
    
    def __call__(self, state: GraphState) -> GraphState:
        """Analyze and optimize the query."""
        logger.info("Analyzing and optimizing query")
        
        question = state["question"]
        strategy = state.get("optimization_strategy", "hybrid")
        
        # Optimize the query
        optimization_result = self.query_rewriter.optimize_query(question, strategy)
        
        # Update state
        state["optimized_queries"] = optimization_result["optimized_queries"]
        state["metadata"]["query_optimization"] = optimization_result["metadata"]
        
        logger.info(f"Generated {len(optimization_result['optimized_queries'])} optimized queries")
        
        return state


class RetrievalNode:
    """Node for retrieving documents from vector store."""
    
    def __init__(self, vector_store: FAISS, k: int = 5):
        self.vector_store = vector_store
        self.k = k
    
    def __call__(self, state: GraphState) -> GraphState:
        """Retrieve documents for all optimized queries."""
        logger.info("Retrieving documents")
        
        optimized_queries = state["optimized_queries"]
        all_documents = []
        
        # Retrieve documents for each optimized query
        for query in optimized_queries:
            try:
                docs = self.vector_store.similarity_search(query, k=self.k)
                all_documents.extend(docs)
                logger.info(f"Retrieved {len(docs)} documents for query: {query[:50]}...")
            except Exception as e:
                logger.error(f"Error retrieving documents for query '{query}': {e}")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_documents = []
        for doc in all_documents:
            content_hash = hash(doc.page_content)
            if content_hash not in seen:
                seen.add(content_hash)
                unique_documents.append(doc)
        
        state["retrieved_documents"] = unique_documents
        state["metadata"]["retrieval_stats"] = {
            "total_queries": len(optimized_queries),
            "total_documents": len(all_documents),
            "unique_documents": len(unique_documents),
            "avg_docs_per_query": len(unique_documents) / max(len(optimized_queries), 1)
        }
        
        logger.info(f"Retrieved {len(unique_documents)} unique documents from {len(optimized_queries)} queries")
        
        return state


class GradingNode:
    """Node for grading document relevance."""
    
    def __init__(self, document_grader: DocumentGrader, relevance_threshold: float = 0.6):
        self.document_grader = document_grader
        self.relevance_threshold = relevance_threshold
    
    def __call__(self, state: GraphState) -> GraphState:
        """Grade all retrieved documents for relevance."""
        logger.info("Grading document relevance")
        
        question = state["question"]
        documents = state["retrieved_documents"]
        
        # Grade all documents
        graded_docs = []
        for doc in documents:
            try:
                grading_result = self.document_grader.grade_relevance(question, doc)
                graded_docs.append((doc, grading_result))
            except Exception as e:
                logger.error(f"Error grading document: {e}")
                # Add document with failed grading
                graded_docs.append((doc, {
                    "relevant": False,
                    "confidence": 0.0,
                    "reasoning": f"Grading failed: {str(e)}",
                    "key_info": "",
                    "document_id": getattr(doc, 'metadata', {}).get('id', 'unknown')
                }))
        
        # Filter relevant documents
        relevant_docs = []
        for doc, grading in graded_docs:
            if grading["relevant"] and grading["confidence"] >= self.relevance_threshold:
                relevant_docs.append(doc)
        
        state["graded_documents"] = graded_docs
        state["relevant_documents"] = relevant_docs
        state["metadata"]["grading_stats"] = {
            "total_documents": len(documents),
            "relevant_documents": len(relevant_docs),
            "relevance_rate": len(relevant_docs) / max(len(documents), 1),
            "avg_confidence": sum(g["confidence"] for _, g in graded_docs) / max(len(graded_docs), 1)
        }
        
        logger.info(f"Graded {len(documents)} documents, {len(relevant_docs)} relevant")
        
        return state


class GenerationNode:
    """Node for generating answers from relevant documents."""
    
    def __init__(self, groq_client, model_name: str = "llama-3.3-70b-versatile"):
        self.groq_client = groq_client
        self.model_name = model_name
    
    def __call__(self, state: GraphState) -> GraphState:
        """Generate answer from relevant documents."""
        logger.info("Generating answer from relevant documents")
        
        question = state["question"]
        relevant_docs = state["relevant_documents"]
        
        if not relevant_docs:
            logger.warning("No relevant documents found")
            state["generation"] = "I couldn't find specific information about that in Prajwal's portfolio. Could you ask about something else, like skills, projects, or experience?"
            return state
        
        # Build context from relevant documents
        context_parts = []
        for i, doc in enumerate(relevant_docs[:5], 1):  # Limit to top 5 documents
            context_parts.append(f"Document {i}:\n{doc.page_content}")
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Generate answer
        system_prompt = """
You are Prajwal's AI portfolio assistant. Use the provided documents to answer questions about Prajwal's professional background.

Guidelines:
- Be specific and detailed when possible
- Reference actual information from the documents
- Synthesize information from multiple documents when relevant
- Be professional and helpful
- If information is not in the documents, say so
"""
        
        prompt = f"""
Context from Prajwal's portfolio documents:
{context}

Question: {question}

Please provide a comprehensive answer based on the documents above."""
        
        try:
            response = self.groq_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            generation = response.choices[0].message.content.strip()
            state["generation"] = generation
            
            logger.info(f"Generated answer: {generation[:100]}...")
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            state["generation"] = "I encountered an error while generating the answer. Please try again."
        
        return state


class HallucinationCheckNode:
    """Node for checking generated answers for hallucinations."""
    
    def __init__(self, document_grader: DocumentGrader):
        self.document_grader = document_grader
    
    def __call__(self, state: GraphState) -> GraphState:
        """Check generated answer for hallucinations."""
        logger.info("Checking for hallucinations")
        
        generation = state["generation"]
        relevant_docs = state["relevant_documents"]
        
        if not relevant_docs:
            logger.info("No documents to check against")
            return state
        
        try:
            # Check hallucination
            hallucination_result = self.document_grader.check_hallucination(
                generation, relevant_docs
            )
            
            state["metadata"]["hallucination_check"] = hallucination_result
            
            # If hallucinations detected, mark for potential rewrite
            if hallucination_result["hallucination_detected"]:
                logger.warning("Hallucinations detected in generated answer")
                state["metadata"]["needs_rewrite"] = True
            else:
                logger.info("No hallucinations detected")
                state["metadata"]["needs_rewrite"] = False
                
        except Exception as e:
            logger.error(f"Error checking hallucinations: {e}")
            state["metadata"]["hallucination_check"] = {
                "supported": True,
                "confidence": 0.5,
                "issues": [f"Hallucination check failed: {str(e)}"],
                "hallucination_detected": False
            }
        
        return state


class RewriteDecisionNode:
    """Node for deciding whether to rewrite the query."""
    
    def __init__(self, max_rewrites: int = 2):
        self.max_rewrites = max_rewrites
    
    def __call__(self, state: GraphState) -> str:
        """Decide next step: end, rewrite, or continue."""
        logger.info("Making rewrite decision")
        
        rewrite_count = state.get("rewrite_count", 0)
        relevant_docs = state["relevant_documents"]
        
        # Check if we need to rewrite
        needs_rewrite = state["metadata"].get("needs_rewrite", False)
        
        # Decision logic
        if rewrite_count >= self.max_rewrites:
            logger.info(f"Max rewrites ({self.max_rewrites}) reached, ending")
            return "end"
        
        if not relevant_docs:
            logger.info("No relevant documents found, rewriting query")
            return "rewrite"
        
        if needs_rewrite:
            logger.info("Hallucinations detected, rewriting query")
            return "rewrite"
        
        logger.info("Satisfactory results, ending")
        return "end"