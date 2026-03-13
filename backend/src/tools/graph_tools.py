from typing import Dict, Any, List, Optional, Callable
from abc import ABC, abstractmethod
import logging
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class BaseTool(ABC):
    """Base class for all agent tools."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.usage_count = 0
        self.last_used = None
    
    @abstractmethod
    def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute the tool with given parameters."""
        pass
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get tool metadata."""
        return {
            "name": self.name,
            "description": self.description,
            "usage_count": self.usage_count,
            "last_used": self.last_used.isoformat() if self.last_used else None
        }


class PortfolioSearchTool(BaseTool):
    """Tool for searching Prajwal's portfolio documents."""
    
    def __init__(self, vector_store, embedding_model):
        super().__init__(
            name="portfolio_search",
            description="Search Prajwal's portfolio documents for relevant information about skills, projects, and experience"
        )
        self.vector_store = vector_store
        self.embedding_model = embedding_model
    
    def execute(self, query: str, k: int = 3, **kwargs) -> Dict[str, Any]:
        """Search portfolio documents."""
        try:
            self.usage_count += 1
            self.last_used = datetime.now()
            
            if not self.vector_store:
                return {
                    "success": False,
                    "error": "Vector store not available",
                    "documents": [],
                    "query": query
                }
            
            # Perform similarity search
            documents = self.vector_store.similarity_search(query, k=k)
            
            # Format results
            formatted_docs = []
            for i, doc in enumerate(documents):
                formatted_docs.append({
                    "id": i,
                    "content": doc.page_content[:500],  # Limit content
                    "metadata": getattr(doc, 'metadata', {})
                })
            
            logger.info(f"Portfolio search found {len(formatted_docs)} documents for query: {query[:50]}")
            
            return {
                "success": True,
                "documents": formatted_docs,
                "query": query,
                "count": len(formatted_docs)
            }
            
        except Exception as e:
            logger.error(f"Error in portfolio search: {e}")
            return {
                "success": False,
                "error": str(e),
                "documents": [],
                "query": query
            }


class WebSearchTool(BaseTool):
    """Tool for searching the web for current information."""
    
    def __init__(self):
        super().__init__(
            name="web_search",
            description="Search the web for current information about technologies, trends, and general knowledge"
        )
    
    def execute(self, query: str, max_results: int = 3, **kwargs) -> Dict[str, Any]:
        """Perform web search."""
        try:
            self.usage_count += 1
            self.last_used = datetime.now()
            
            # Note: This is a mock implementation
            # In production, integrate with actual web search APIs like SerpAPI, Bing Search, etc.
            
            logger.info(f"Web search would be performed for: {query[:50]}")
            
            # Mock results for demonstration
            mock_results = [
                {
                    "title": f"Information about {query}",
                    "snippet": f"This is a mock search result for '{query}'. In production, this would contain real web search results.",
                    "url": "https://example.com"
                }
            ]
            
            return {
                "success": True,
                "results": mock_results[:max_results],
                "query": query,
                "count": len(mock_results[:max_results]),
                "note": "This is a mock implementation. Integrate with real web search API for production."
            }
            
        except Exception as e:
            logger.error(f"Error in web search: {e}")
            return {
                "success": False,
                "error": str(e),
                "results": [],
                "query": query
            }


class DataAnalysisTool(BaseTool):
    """Tool for analyzing data and extracting insights."""
    
    def __init__(self):
        super().__init__(
            name="data_analysis",
            description="Analyze provided data, documents, or information to extract key insights and patterns"
        )
    
    def execute(self, data: Any, analysis_type: str = "summary", **kwargs) -> Dict[str, Any]:
        """Analyze provided data."""
        try:
            self.usage_count += 1
            self.last_used = datetime.now()
            
            # Convert data to string if needed
            if isinstance(data, (list, dict)):
                data_str = json.dumps(data, indent=2)
            else:
                data_str = str(data)
            
            # Mock analysis based on type
            if analysis_type == "summary":
                result = self._summarize_data(data_str)
            elif analysis_type == "keywords":
                result = self._extract_keywords(data_str)
            elif analysis_type == "sentiment":
                result = self._analyze_sentiment(data_str)
            else:
                result = self._general_analysis(data_str)
            
            logger.info(f"Data analysis completed for {analysis_type}")
            
            return {
                "success": True,
                "analysis": result,
                "type": analysis_type,
                "data_size": len(data_str)
            }
            
        except Exception as e:
            logger.error(f"Error in data analysis: {e}")
            return {
                "success": False,
                "error": str(e),
                "analysis": None,
                "type": analysis_type
            }
    
    def _summarize_data(self, data: str) -> Dict[str, Any]:
        """Generate summary of data."""
        words = data.split()
        return {
            "summary": f"Data contains {len(words)} words with key themes that would be identified in production.",
            "word_count": len(words),
            "key_points": ["Point 1", "Point 2", "Point 3"]  # Mock key points
        }
    
    def _extract_keywords(self, data: str) -> Dict[str, Any]:
        """Extract keywords from data."""
        # Simple keyword extraction (mock)
        words = data.lower().split()
        common_words = ["the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"]
        keywords = [word for word in words if word not in common_words and len(word) > 3]
        
        return {
            "keywords": list(set(keywords))[:10],  # Top 10 unique keywords
            "total_words": len(words),
            "keyword_count": len(set(keywords))
        }
    
    def _analyze_sentiment(self, data: str) -> Dict[str, Any]:
        """Analyze sentiment of data."""
        # Mock sentiment analysis
        return {
            "sentiment": "neutral",
            "confidence": 0.8,
            "note": "In production, this would use actual sentiment analysis models"
        }
    
    def _general_analysis(self, data: str) -> Dict[str, Any]:
        """General data analysis."""
        return {
            "type": "general",
            "length": len(data),
            "structure": "text" if isinstance(data, str) else "structured"
        }


class ResponseGenerationTool(BaseTool):
    """Tool for generating final responses based on gathered information."""
    
    def __init__(self, groq_client):
        super().__init__(
            name="generate_response",
            description="Generate a comprehensive response based on gathered information and context"
        )
        self.groq_client = groq_client
    
    def execute(self, query: str, context: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Generate response based on context."""
        try:
            self.usage_count += 1
            self.last_used = datetime.now()
            
            # Build context for response generation
            context_text = self._format_context(context)
            
            prompt = f"""
Based on the following information, generate a comprehensive and accurate response to the user's query.

User Query: "{query}"

Context and Gathered Information:
{context_text}

Instructions:
- Be specific and detailed when possible
- Reference actual information from the context
- Be professional and helpful
- If information is incomplete, state what you can and note limitations
- Structure the response clearly

Response:
"""
            
            response = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are generating a response based on gathered information."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            generated_response = response.choices[0].message.content.strip()
            
            logger.info(f"Response generated for query: {query[:50]}")
            
            return {
                "success": True,
                "response": generated_response,
                "query": query,
                "context_used": list(context.keys())
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {
                "success": False,
                "error": str(e),
                "response": "I encountered an error generating the response.",
                "query": query
            }
    
    def _format_context(self, context: Dict[str, Any]) -> str:
        """Format context for prompt."""
        formatted_parts = []
        
        for key, value in context.items():
            if isinstance(value, dict) and "success" in value:
                if value["success"]:
                    formatted_parts.append(f"{key.replace('_', ' ').title()}:\n{json.dumps(value, indent=2)}")
                else:
                    formatted_parts.append(f"{key.replace('_', ' ').title()}: Failed - {value.get('error', 'Unknown error')}")
            else:
                formatted_parts.append(f"{key.replace('_', ' ').title()}:\n{str(value)}")
        
        return "\n\n".join(formatted_parts)


class ToolRegistry:
    """Registry for managing agent tools."""
    
    def __init__(self):
        self.tools: Dict[str, BaseTool] = {}
        self.tool_descriptions = []
    
    def register_tool(self, tool: BaseTool):
        """Register a tool."""
        self.tools[tool.name] = tool
        self.tool_descriptions.append({
            "name": tool.name,
            "description": tool.description
        })
        logger.info(f"Registered tool: {tool.name}")
    
    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get a tool by name."""
        return self.tools.get(name)
    
    def list_tools(self) -> List[Dict[str, str]]:
        """List all available tools."""
        return self.tool_descriptions.copy()
    
    def get_tool_metadata(self) -> Dict[str, Any]:
        """Get metadata for all tools."""
        return {
            name: tool.get_metadata() 
            for name, tool in self.tools.items()
        }
    
    def execute_tool(self, name: str, **kwargs) -> Dict[str, Any]:
        """Execute a tool by name."""
        tool = self.get_tool(name)
        if not tool:
            return {
                "success": False,
                "error": f"Tool '{name}' not found"
            }
        
        return tool.execute(**kwargs)