from typing import List, Dict, Any, Optional, Tuple
import logging
from groq import Groq
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class DocumentGrader:
    """Advanced document relevance grading system."""
    
    def __init__(self, groq_client: Groq):
        self.client = groq_client
        
        # Grading prompts for different aspects
        self.relevance_prompt = """
You are a document relevance expert. Evaluate if the provided document contains information that answers the user's question.

Question: "{question}"

Document Content:
{document_content}

Task: Determine if this document contains relevant information to answer the question.

Respond in this format:
RELEVANT: [yes/no]
CONFIDENCE: [0.0-1.0]
REASONING: [brief explanation of why it's relevant or not]
KEY_INFO: [specific information from document that answers the question, if any]

Guidelines:
- Be strict but fair in your evaluation
- Consider partial relevance (document doesn't need to completely answer the question)
- Look for specific facts, examples, or explanations that address the question
- If document is about a different topic, mark as not relevant
"""
        
        self.hallucination_prompt = """
You are a fact-checking expert. Compare the generated answer with the provided documents to check for hallucinations.

Generated Answer: "{generated_answer}"

Document Content:
{document_content}

Task: Check if the generated answer is supported by the document content.

Respond in this format:
SUPPORTED: [yes/no]
CONFIDENCE: [0.0-1.0]
ISSUES: [list any unsupported claims or hallucinations]
SUGGESTION: [brief suggestion for improvement if needed]

Guidelines:
- Be thorough in checking factual alignment
- Identify any claims not supported by documents
- Consider paraphrasing vs. factual contradictions
- Suggest improvements if hallucinations are found
"""
        
        self.quality_prompt = """
You are a document quality assessor. Evaluate the quality and usefulness of this document for answering questions.

Document Content:
{document_content}

Task: Assess the document's quality for use in question answering.

Respond in this format:
QUALITY_SCORE: [1-10]
COMPLETENESS: [high/medium/low]
CLARITY: [high/medium/low]
TECHNICAL_ACCURACY: [high/medium/low]
STRENGTHS: [list key strengths]
WEAKNESSES: [list key weaknesses]
OVERALL_ASSESSMENT: [brief overall assessment]

Guidelines:
- Consider factual accuracy, completeness, and clarity
- Evaluate technical depth and usefulness
- Identify both strengths and areas for improvement
- Provide constructive feedback
"""
    
    def grade_relevance(self, question: str, document: Document) -> Dict[str, Any]:
        """
        Grade document relevance to the question.
        
        Args:
            question: User's question
            document: Document to grade
            
        Returns:
            Relevance grading results
        """
        try:
            document_content = document.page_content[:2000]  # Limit content length
            prompt = self.relevance_prompt.format(
                question=question,
                document_content=document_content
            )
            
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are a document relevance expert."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=200
            )
            
            grading_text = response.choices[0].message.content.strip()
            
            # Parse grading response
            result = {
                "relevant": False,
                "confidence": 0.0,
                "reasoning": "",
                "key_info": "",
                "document_id": getattr(document, 'metadata', {}).get('id', 'unknown')
            }
            
            for line in grading_text.split('\n'):
                line = line.strip()
                if line.startswith('RELEVANT:'):
                    result["relevant"] = 'yes' in line.lower()
                elif line.startswith('CONFIDENCE:'):
                    try:
                        result["confidence"] = float(line.split(':', 1)[1].strip())
                    except ValueError:
                        pass
                elif line.startswith('REASONING:'):
                    result["reasoning"] = line.split(':', 1)[1].strip()
                elif line.startswith('KEY_INFO:'):
                    result["key_info"] = line.split(':', 1)[1].strip()
            
            logger.info(f"Graded document relevance: {result['relevant']} (confidence: {result['confidence']})")
            return result
            
        except Exception as e:
            logger.error(f"Error grading document relevance: {e}")
            return {
                "relevant": False,
                "confidence": 0.0,
                "reasoning": f"Error during grading: {str(e)}",
                "key_info": "",
                "document_id": getattr(document, 'metadata', {}).get('id', 'unknown')
            }
    
    def check_hallucination(self, generated_answer: str, documents: List[Document]) -> Dict[str, Any]:
        """
        Check for hallucinations by comparing generated answer with documents.
        
        Args:
            generated_answer: The answer generated by the LLM
            documents: Source documents used for generation
            
        Returns:
            Hallucination check results
        """
        try:
            # Combine all document contents
            combined_content = "\n\n---\n\n".join([doc.page_content[:1500] for doc in documents])
            
            prompt = self.hallucination_prompt.format(
                generated_answer=generated_answer[:1000],
                document_content=combined_content
            )
            
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are a fact-checking expert."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=300
            )
            
            check_text = response.choices[0].message.content.strip()
            
            # Parse hallucination check
            result = {
                "supported": True,
                "confidence": 0.0,
                "issues": [],
                "suggestion": "",
                "hallucination_detected": False
            }
            
            for line in check_text.split('\n'):
                line = line.strip()
                if line.startswith('SUPPORTED:'):
                    result["supported"] = 'yes' in line.lower()
                elif line.startswith('CONFIDENCE:'):
                    try:
                        result["confidence"] = float(line.split(':', 1)[1].strip())
                    except ValueError:
                        pass
                elif line.startswith('ISSUES:'):
                    issues_text = line.split(':', 1)[1].strip()
                    if issues_text and issues_text.lower() != 'none':
                        result["issues"] = [issues_text]
                        result["hallucination_detected"] = True
                elif line.startswith('SUGGESTION:'):
                    result["suggestion"] = line.split(':', 1)[1].strip()
            
            logger.info(f"Hallucination check: supported={result['supported']}, detected={result['hallucination_detected']}")
            return result
            
        except Exception as e:
            logger.error(f"Error checking hallucinations: {e}")
            return {
                "supported": True,
                "confidence": 0.5,
                "issues": [f"Error during hallucination check: {str(e)}"],
                "suggestion": "Unable to verify due to error",
                "hallucination_detected": False
            }
    
    def grade_quality(self, document: Document) -> Dict[str, Any]:
        """
        Grade overall document quality.
        
        Args:
            document: Document to grade
            
        Returns:
            Quality assessment results
        """
        try:
            document_content = document.page_content[:2000]
            prompt = self.quality_prompt.format(document_content=document_content)
            
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are a document quality assessor."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=300
            )
            
            quality_text = response.choices[0].message.content.strip()
            
            # Parse quality assessment
            result = {
                "quality_score": 5,
                "completeness": "medium",
                "clarity": "medium", 
                "technical_accuracy": "medium",
                "strengths": [],
                "weaknesses": [],
                "overall_assessment": ""
            }
            
            for line in quality_text.split('\n'):
                line = line.strip()
                if line.startswith('QUALITY_SCORE:'):
                    try:
                        result["quality_score"] = int(line.split(':', 1)[1].strip())
                    except ValueError:
                        pass
                elif line.startswith('COMPLETENESS:'):
                    result["completeness"] = line.split(':', 1)[1].strip().lower()
                elif line.startswith('CLARITY:'):
                    result["clarity"] = line.split(':', 1)[1].strip().lower()
                elif line.startswith('TECHNICAL_ACCURACY:'):
                    result["technical_accuracy"] = line.split(':', 1)[1].strip().lower()
                elif line.startswith('STRENGTHS:'):
                    strengths_text = line.split(':', 1)[1].strip()
                    if strengths_text and strengths_text.lower() != 'none':
                        result["strengths"] = [s.strip() for s in strengths_text.split(',')]
                elif line.startswith('WEAKNESSES:'):
                    weaknesses_text = line.split(':', 1)[1].strip()
                    if weaknesses_text and weaknesses_text.lower() != 'none':
                        result["weaknesses"] = [w.strip() for w in weaknesses_text.split(',')]
                elif line.startswith('OVERALL_ASSESSMENT:'):
                    result["overall_assessment"] = line.split(':', 1)[1].strip()
            
            logger.info(f"Graded document quality: {result['quality_score']}/10")
            return result
            
        except Exception as e:
            logger.error(f"Error grading document quality: {e}")
            return {
                "quality_score": 5,
                "completeness": "unknown",
                "clarity": "unknown",
                "technical_accuracy": "unknown",
                "strengths": [],
                "weaknesses": [f"Error during quality grading: {str(e)}"],
                "overall_assessment": "Unable to assess due to error"
            }
    
    def filter_relevant_documents(self, question: str, documents: List[Document], 
                                  relevance_threshold: float = 0.6) -> List[Tuple[Document, Dict[str, Any]]]:
        """
        Filter documents based on relevance grading.
        
        Args:
            question: User's question
            documents: List of documents to grade
            relevance_threshold: Minimum relevance score to keep document
            
        Returns:
            List of (document, grading_results) tuples for relevant documents
        """
        logger.info(f"Grading relevance for {len(documents)} documents")
        
        relevant_docs = []
        
        for doc in documents:
            grading_result = self.grade_relevance(question, doc)
            
            if grading_result["relevant"] and grading_result["confidence"] >= relevance_threshold:
                relevant_docs.append((doc, grading_result))
                logger.info(f"Document {grading_result['document_id']} passed relevance filter")
            else:
                logger.info(f"Document {grading_result['document_id']} filtered out: {grading_result['reasoning']}")
        
        logger.info(f"Filtered to {len(relevant_docs)} relevant documents")
        return relevant_docs
    
    def grade_document_batch(self, question: str, documents: List[Document]) -> List[Dict[str, Any]]:
        """
        Grade a batch of documents efficiently.
        
        Args:
            question: User's question
            documents: List of documents to grade
            
        Returns:
            List of grading results
        """
        results = []
        
        for doc in documents:
            grading_result = self.grade_relevance(question, doc)
            quality_result = self.grade_quality(doc)
            
            combined_result = {
                **grading_result,
                "quality_assessment": quality_result,
                "combined_score": (grading_result["confidence"] * quality_result["quality_score"]) / 10
            }
            
            results.append(combined_result)
        
        return results