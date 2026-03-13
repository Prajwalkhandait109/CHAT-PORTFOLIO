from typing import List, Optional, Dict, Any
import logging
from groq import Groq

logger = logging.getLogger(__name__)


class QueryRewriter:
    """Advanced query rewriter for optimizing retrieval performance."""
    
    def __init__(self, groq_client: Groq):
        self.client = groq_client
        
        # Query optimization prompts
        self.decomposition_prompt = """
You are a query decomposition expert. Break down complex queries into simpler sub-queries for better information retrieval.

Original Query: "{query}"

Task: Decompose this query into 2-4 simpler, focused sub-queries that would help answer the main question.
Each sub-query should target a specific aspect of the original query.

Respond in this format:
SUB_QUERY_1: [focused sub-query 1]
SUB_QUERY_2: [focused sub-query 2]
SUB_QUERY_3: [focused sub-query 3] (if needed)
SUB_QUERY_4: [focused sub-query 4] (if needed)

Guidelines:
- Make sub-queries specific and actionable
- Focus on different aspects of the main query
- Keep sub-queries concise and clear
- Ensure sub-queries are relevant to the original intent
"""
        
        self.hyde_prompt = """
You are a helpful assistant. Generate a hypothetical answer to help with document retrieval.

Query: "{query}"

Task: Write a brief hypothetical answer (2-3 sentences) that would perfectly answer this query.
This hypothetical answer will be used to find relevant documents through similarity search.

Guidelines:
- Make the answer specific and detailed
- Include key terms and concepts that would likely appear in relevant documents
- Keep it concise but comprehensive
- Focus on factual information that would answer the query

Hypothetical Answer:
"""
        
        self.expansion_prompt = """
You are a query expansion expert. Expand queries with synonyms and related terms to improve retrieval recall.

Original Query: "{query}"

Task: Generate 3-5 alternative phrasings or expansions of this query that capture the same intent.

Respond in this format:
EXPANSION_1: [alternative phrasing 1]
EXPANSION_2: [alternative phrasing 2]
EXPANSION_3: [alternative phrasing 3]
EXPANSION_4: [alternative phrasing 4] (if needed)
EXPANSION_5: [alternative phrasing 5] (if needed)

Guidelines:
- Use synonyms and related terms
- Maintain the original intent
- Consider different ways the same information might be phrased
- Include technical terms if relevant
"""
    
    def decompose_query(self, query: str) -> List[str]:
        """
        Decompose complex queries into simpler sub-queries.
        
        Args:
            query: Original complex query
            
        Returns:
            List of decomposed sub-queries
        """
        try:
            prompt = self.decomposition_prompt.format(query=query)
            
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are a query decomposition expert."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=300
            )
            
            decomposition_text = response.choices[0].message.content.strip()
            sub_queries = []
            
            # Parse sub-queries from response
            for line in decomposition_text.split('\n'):
                line = line.strip()
                if line.startswith('SUB_QUERY_') and ':' in line:
                    sub_query = line.split(':', 1)[1].strip()
                    if sub_query and len(sub_query) > 10:  # Filter out empty/short queries
                        sub_queries.append(sub_query)
            
            # If no sub-queries were parsed, return original query
            if not sub_queries:
                logger.warning(f"Failed to decompose query: {query}")
                return [query]
            
            logger.info(f"Decomposed query into {len(sub_queries)} sub-queries")
            return sub_queries
            
        except Exception as e:
            logger.error(f"Error decomposing query: {e}")
            return [query]  # Fallback to original query
    
    def generate_hypothetical_answer(self, query: str) -> str:
        """
        Generate a hypothetical answer for HyDE (Hypothetical Document Embeddings) approach.
        
        Args:
            query: User query
            
        Returns:
            Hypothetical answer that would answer the query
        """
        try:
            prompt = self.hyde_prompt.format(query=query)
            
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant generating hypothetical answers."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=200
            )
            
            hypothetical_answer = response.choices[0].message.content.strip()
            
            # Clean up the response
            if "Hypothetical Answer:" in hypothetical_answer:
                hypothetical_answer = hypothetical_answer.split("Hypothetical Answer:")[1].strip()
            
            logger.info(f"Generated hypothetical answer for HyDE: {hypothetical_answer[:100]}...")
            return hypothetical_answer
            
        except Exception as e:
            logger.error(f"Error generating hypothetical answer: {e}")
            return query  # Fallback to original query
    
    def expand_query(self, query: str) -> List[str]:
        """
        Expand query with synonyms and alternative phrasings.
        
        Args:
            query: Original query
            
        Returns:
            List of expanded query variations
        """
        try:
            prompt = self.expansion_prompt.format(query=query)
            
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are a query expansion expert."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4,
                max_tokens=250
            )
            
            expansion_text = response.choices[0].message.content.strip()
            expansions = []
            
            # Parse expansions from response
            for line in expansion_text.split('\n'):
                line = line.strip()
                if line.startswith('EXPANSION_') and ':' in line:
                    expansion = line.split(':', 1)[1].strip()
                    if expansion and len(expansion) > 5:
                        expansions.append(expansion)
            
            # If no expansions were parsed, return original query
            if not expansions:
                logger.warning(f"Failed to expand query: {query}")
                return [query]
            
            logger.info(f"Expanded query into {len(expansions)} variations")
            return expansions
            
        except Exception as e:
            logger.error(f"Error expanding query: {e}")
            return [query]  # Fallback to original query
    
    def optimize_query(self, query: str, strategy: str = "hybrid") -> Dict[str, Any]:
        """
        Optimize query using multiple strategies.
        
        Args:
            query: Original user query
            strategy: Optimization strategy ("decompose", "hyde", "expand", "hybrid")
            
        Returns:
            Dictionary with optimized queries and metadata
        """
        logger.info(f"Optimizing query with strategy: {strategy}")
        
        result = {
            "original_query": query,
            "strategy": strategy,
            "optimized_queries": [],
            "metadata": {}
        }
        
        if strategy == "decompose":
            sub_queries = self.decompose_query(query)
            result["optimized_queries"] = sub_queries
            result["metadata"]["decomposition_count"] = len(sub_queries)
            
        elif strategy == "hyde":
            hypothetical_answer = self.generate_hypothetical_answer(query)
            result["optimized_queries"] = [query, hypothetical_answer]
            result["metadata"]["hyde_answer"] = hypothetical_answer
            
        elif strategy == "expand":
            expansions = self.expand_query(query)
            result["optimized_queries"] = [query] + expansions
            result["metadata"]["expansion_count"] = len(expansions)
            
        elif strategy == "hybrid":
            # Combine multiple strategies
            sub_queries = self.decompose_query(query)
            hypothetical_answer = self.generate_hypothetical_answer(query)
            expansions = self.expand_query(query)
            
            # Combine all optimizations, ensuring original query is first
            all_queries = [query]
            if sub_queries:
                all_queries.extend(sub_queries[:2])  # Limit to avoid too many queries
            if hypothetical_answer and hypothetical_answer != query:
                all_queries.append(hypothetical_answer)
            if expansions:
                all_queries.extend(expansions[:2])  # Limit expansions
            
            result["optimized_queries"] = list(dict.fromkeys(all_queries))  # Remove duplicates while preserving order
            result["metadata"] = {
                "decomposition_count": len(sub_queries),
                "hyde_generated": hypothetical_answer != query,
                "expansion_count": len(expansions),
                "total_optimized": len(result["optimized_queries"])
            }
            
        else:
            logger.warning(f"Unknown optimization strategy: {strategy}")
            result["optimized_queries"] = [query]
        
        return result