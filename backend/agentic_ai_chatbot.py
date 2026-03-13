from groq import Groq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from typing import Dict, Any, Optional, List
import logging
import uuid

from src.rag.router import QueryRouter
from src.rag.graph_builder import AdvancedRAGPipeline, SimpleRAGPipeline
from src.rag.reAct_agent import ReActAgent
from src.tools.graph_tools import (
    PortfolioSearchTool, WebSearchTool, DataAnalysisTool, 
    ResponseGenerationTool, ToolRegistry
)
from src.memory.agent_memory import AgentMemory
from src.models.route_identifier import ClassificationResult

logger = logging.getLogger(__name__)


class AgenticAIChatbot:
    """Advanced chatbot with ReAct agent, intelligent routing, and multi-stage RAG."""
    
    def __init__(self, groq_api_key: str, use_advanced_rag: bool = True, enable_agent: bool = True):
        self.client = Groq(api_key=groq_api_key)
        self.use_advanced_rag = use_advanced_rag
        self.enable_agent = enable_agent
        
        # Initialize routing system
        self.router = QueryRouter(self.client)
        
        # Initialize vector store for portfolio queries
        self.embedding = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        try:
            self.db = FAISS.load_local("db", self.embedding, allow_dangerous_deserialization=True)
            logger.info("Vector store loaded successfully")
            
            # Initialize RAG pipelines
            if use_advanced_rag:
                self.advanced_rag = AdvancedRAGPipeline(self.client, self.db)
                self.simple_rag = SimpleRAGPipeline(self.client, self.db)
                logger.info("Advanced RAG pipeline initialized")
            else:
                self.advanced_rag = None
                self.simple_rag = SimpleRAGPipeline(self.client, self.db)
                logger.info("Simple RAG pipeline initialized")
                
        except Exception as e:
            logger.error(f"Failed to load vector store: {e}")
            self.db = None
            self.advanced_rag = None
            self.simple_rag = None
        
        # Initialize agent system if enabled
        if enable_agent:
            self._initialize_agent_system()
        else:
            self.reAct_agent = None
            self.tool_registry = None
            self.agent_memory = None
        
        # Register handlers
        self._register_handlers()
    
    def _initialize_agent_system(self):
        """Initialize the ReAct agent system with tools and memory."""
        try:
            # Initialize memory
            self.agent_memory = AgentMemory(max_conversation_history=10, max_memory_entries=50)
            
            # Initialize tool registry
            self.tool_registry = ToolRegistry()
            
            # Register tools
            if self.db:
                portfolio_tool = PortfolioSearchTool(self.db, self.embedding)
                self.tool_registry.register_tool(portfolio_tool)
            
            web_tool = WebSearchTool()
            self.tool_registry.register_tool(web_tool)
            
            data_analysis_tool = DataAnalysisTool()
            self.tool_registry.register_tool(data_analysis_tool)
            
            response_tool = ResponseGenerationTool(self.client)
            self.tool_registry.register_tool(response_tool)
            
            # Initialize ReAct agent
            self.reAct_agent = ReActAgent(self.client, max_steps=5)
            
            logger.info("Agent system initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize agent system: {e}")
            self.reAct_agent = None
            self.tool_registry = None
            self.agent_memory = None
    
    def _register_handlers(self):
        """Register all query handlers."""
        self.router.register_handler("portfolio_handler", self._handle_portfolio_query)
        self.router.register_handler("technical_handler", self._handle_technical_query)
        self.router.register_handler("general_handler", self._handle_general_query)
        self.router.register_handler("greeting_handler", self._handle_greeting_query)
        self.router.register_handler("goodbye_handler", self._handle_goodbye_query)
        self.router.register_handler("clarification_handler", self._handle_clarification_query)
        self.router.register_handler("scope_handler", self._handle_scope_query)
        
        if self.enable_agent:
            self.router.register_handler("agent_handler", self._handle_agent_query)
    
    def _handle_portfolio_query(self, query: str, classification: ClassificationResult, 
                               should_use_rag: bool, context_required: bool, **kwargs) -> str:
        """Handle portfolio-related queries using RAG."""
        if not self.db:
            return "I apologize, but I don't have access to portfolio information right now."
        
        try:
            if self.use_advanced_rag and self.advanced_rag:
                # Use advanced RAG pipeline
                logger.info("Using advanced RAG pipeline for portfolio query")
                result = self.advanced_rag.invoke(query, optimization_strategy="hybrid")
                
                # Log detailed metrics
                metadata = result.get("metadata", {})
                logger.info(f"Advanced RAG completed: {result.get('relevant_documents', 0)} relevant docs")
                
                return result["answer"]
            else:
                # Use simple RAG pipeline
                logger.info("Using simple RAG pipeline for portfolio query")
                result = self.simple_rag.simple_retrieve_and_generate(query, k=5)
                
                logger.info(f"Simple RAG completed: {result.get('relevant_documents', 0)} relevant docs")
                return result["answer"]
                
        except Exception as e:
            logger.error(f"Error in portfolio handler: {e}")
            return "I encountered an error while searching Prajwal's portfolio. Please try again."
    
    def _handle_agent_query(self, query: str, classification: ClassificationResult,
                           should_use_rag: bool, context_required: bool, **kwargs) -> str:
        """Handle queries using the ReAct agent."""
        if not self.reAct_agent or not self.tool_registry:
            return "Agent system is not available. Using standard RAG instead."
        
        try:
            # Start new session if needed
            session_id = kwargs.get("session_id", str(uuid.uuid4()))
            if not self.agent_memory.current_session_id:
                self.agent_memory.start_session(session_id)
            
            # Get conversation context
            conversation_context = self.agent_memory.get_conversation_context(session_id, last_n=3)
            
            # Build agent context
            agent_context = {
                "conversation_history": conversation_context,
                "available_tools": self.tool_registry.list_tools(),
                "user_query": query,
                "classification": classification.category.value,
                "confidence": classification.confidence
            }
            
            # Run ReAct agent
            agent_result = self.reAct_agent.run(query, agent_context)
            
            # Store the interaction
            final_answer = agent_result.get("final_answer", "No answer generated")
            self.agent_memory.add_conversation_turn(
                user_query=query,
                agent_response=final_answer,
                metadata={
                    "agent_used": True,
                    "steps_taken": agent_result.get("steps_taken", 0),
                    "confidence": agent_result.get("confidence", 0),
                    "thoughts_count": len(agent_result.get("thoughts", [])),
                    "tools_used": self._extract_tools_used(agent_result)
                }
            )
            
            logger.info(f"ReAct agent completed in {agent_result.get('steps_taken', 0)} steps")
            return final_answer
            
        except Exception as e:
            logger.error(f"Error in agent handler: {e}")
            return "I encountered an error with the agent system. Please try again."
    
    def _extract_tools_used(self, agent_result: Dict[str, Any]) -> List[str]:
        """Extract tools used from agent result."""
        tools_used = []
        for thought in agent_result.get("thoughts", []):
            action = thought.get("action", "")
            if action and action != "reason":
                tools_used.append(action)
        return list(set(tools_used))
    
    def _handle_technical_query(self, query: str, classification: ClassificationResult, 
                               should_use_rag: bool, context_required: bool, **kwargs) -> str:
        """Handle technical questions using general knowledge."""
        system_prompt = """
You are a knowledgeable technical assistant. Answer technical questions clearly and accurately.

Guidelines:
- Provide accurate technical information
- Use examples when helpful
- Be concise but comprehensive
- If unsure about something, say so
"""
        
        try:
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                temperature=0.7,
                max_tokens=400
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error in technical handler: {e}")
            return "I encountered an error while processing your technical question. Please try again."
    
    def _handle_general_query(self, query: str, classification: ClassificationResult, 
                             should_use_rag: bool, context_required: bool, **kwargs) -> str:
        """Handle general knowledge questions."""
        system_prompt = """
You are a helpful AI assistant. Answer general questions clearly and concisely.

Guidelines:
- Provide accurate information
- Be helpful and friendly
- Keep responses concise but informative
- If the question is too broad, ask for clarification
"""
        
        try:
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                temperature=0.7,
                max_tokens=300
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error in general handler: {e}")
            return "I encountered an error while processing your question. Please try again."
    
    def _handle_greeting_query(self, query: str, classification: ClassificationResult, 
                              should_use_rag: bool, context_required: bool, **kwargs) -> str:
        """Handle greeting queries."""
        return "Hello! I'm Prajwal's AI portfolio assistant. I can help you learn about Prajwal's skills, projects, and professional experience. What would you like to know?"
    
    def _handle_goodbye_query(self, query: str, classification: ClassificationResult, 
                             should_use_rag: bool, context_required: bool, **kwargs) -> str:
        """Handle goodbye queries."""
        return "Thank you for your interest in Prajwal's portfolio! Feel free to ask more questions anytime. Goodbye!"
    
    def _handle_clarification_query(self, query: str, classification: ClassificationResult, 
                                   should_use_rag: bool, context_required: bool, **kwargs) -> str:
        """Handle unclear queries that need clarification."""
        return "I'm not quite sure what you're asking. Could you please rephrase your question? I can help you learn about Prajwal's skills, projects, and experience."
    
    def _handle_scope_query(self, query: str, classification: ClassificationResult, 
                           should_use_rag: bool, context_required: bool, **kwargs) -> str:
        """Handle out-of-scope queries."""
        return "I can only answer questions about Prajwal's portfolio, including skills, projects, and experience. Please ask something related to Prajwal's professional background."
    
    def ask(self, query: str, session_context: Optional[Dict[str, Any]] = None, 
            use_advanced_rag: Optional[bool] = None, enable_agent: Optional[bool] = None) -> Dict[str, Any]:
        """
        Process a query through the intelligent routing system with optional agent support.
        
        Args:
            query: The user's question
            session_context: Optional session context
            use_advanced_rag: Override default RAG mode
            enable_agent: Override default agent mode
            
        Returns:
            Dictionary with response and metadata
        """
        try:
            # Use provided modes or defaults
            rag_mode = use_advanced_rag if use_advanced_rag is not None else self.use_advanced_rag
            agent_mode = enable_agent if enable_agent is not None else self.enable_agent
            
            # Route the query
            route_decision = self.router.route(query, session_context)
            
            # Check if we should use agent for this query
            if (agent_mode and self.reAct_agent and 
                route_decision.classification.category.value in ["PORTFOLIO", "TECHNICAL", "GENERAL"] and
                route_decision.classification.confidence > 0.6):
                
                # Use agent for complex queries
                logger.info("Using ReAct agent for query processing")
                result = self.router.execute_route(route_decision)
                
                # Add agent-specific metadata
                result["agent_used"] = True
                result["agent_metadata"] = {
                    "steps_taken": result.get("metadata", {}).get("steps_taken", 0),
                    "tools_used": result.get("metadata", {}).get("tools_used", [])
                }
            else:
                # Use regular routing
                logger.info("Using regular routing for query processing")
                result = self.router.execute_route(route_decision)
                result["agent_used"] = False
            
            # Add system information
            result["system_info"] = {
                "rag_mode": "advanced" if rag_mode else "simple",
                "agent_enabled": agent_mode,
                "classification": route_decision.classification.category.value,
                "confidence": route_decision.classification.confidence
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "response": "I apologize, but I encountered an error processing your request. Please try again.",
                "category": "error",
                "confidence": 0.0,
                "handler_used": "error_handler",
                "error": str(e),
                "agent_used": False
            }
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        stats = {
            "advanced_rag_enabled": self.use_advanced_rag,
            "agent_enabled": self.enable_agent,
            "vector_store_available": self.db is not None,
            "advanced_pipeline_available": self.advanced_rag is not None,
            "simple_pipeline_available": self.simple_rag is not None,
            "agent_system_available": self.reAct_agent is not None
        }
        
        if self.enable_agent and self.agent_memory:
            stats["memory_stats"] = self.agent_memory.get_memory_summary()
            
        if self.enable_agent and self.tool_registry:
            stats["tool_stats"] = self.tool_registry.get_tool_metadata()
            
        if self.enable_agent and self.reAct_agent:
            stats["agent_features"] = [
                "reasoning_and_acting",
                "tool_integration",
                "conversation_memory",
                "multi_step_problem_solving"
            ]
        
        return stats