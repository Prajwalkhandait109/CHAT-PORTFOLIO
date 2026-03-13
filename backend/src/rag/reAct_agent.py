from typing import Dict, Any, List, Optional, TypedDict, Union
from enum import Enum
from dataclasses import dataclass
import json
import logging
from datetime import datetime
from groq import Groq

logger = logging.getLogger(__name__)


class AgentAction(Enum):
    """Available actions for the ReAct agent."""
    REASON = "reason"
    SEARCH_PORTFOLIO = "search_portfolio"
    SEARCH_WEB = "search_web"
    ANALYZE_DATA = "analyze_data"
    GENERATE_RESPONSE = "generate_response"
    CLARIFY = "clarify"
    ESCALATE = "escalate"


@dataclass
class Thought:
    """Represents a reasoning step in the ReAct process."""
    step: int
    content: str
    action: AgentAction
    observation: str
    confidence: float
    timestamp: datetime


@dataclass
class AgentState:
    """State for the ReAct agent."""
    query: str
    thoughts: List[Thought]
    current_action: Optional[AgentAction]
    context: Dict[str, Any]
    max_steps: int
    current_step: int
    completed: bool
    final_answer: Optional[str]
    confidence: float


class ReActAgent:
    """ReAct (Reasoning + Acting) agent implementation."""
    
    def __init__(self, groq_client: Groq, max_steps: int = 5):
        self.client = groq_client
        self.max_steps = max_steps
        
        # ReAct prompt template
        self.react_prompt = """
You are an AI assistant that uses the ReAct (Reasoning + Acting) framework to answer questions.

Current Query: "{query}"

ReAct Process:
1. **Thought**: Analyze what you know and what you need to find out
2. **Action**: Choose an action to take
3. **Observation**: Process the result of your action
4. **Repeat** until you have enough information to answer

Available Actions:
- REASON: Think through the problem logically
- SEARCH_PORTFOLIO: Search Prajwal's portfolio documents
- SEARCH_WEB: Search the web for current information
- ANALYZE_DATA: Analyze provided data or context
- GENERATE_RESPONSE: Generate final answer
- CLARIFY: Ask for clarification
- ESCALATE: Escalate if you cannot answer

Current Context:
{context}

Previous Thoughts:
{thoughts}

Current Step: {current_step}/{max_steps}

Instructions:
1. Start with REASON to analyze the query
2. Use appropriate actions to gather information
3. Build up knowledge through observations
4. Generate final answer when confident
5. Always show your reasoning process

Respond in this format:
THOUGHT: [your reasoning about what to do next]
ACTION: [chosen action]
CONFIDENCE: [0.0-1.0]

If you're ready to answer:
THOUGHT: [final reasoning]
ACTION: GENERATE_RESPONSE
ANSWER: [your final answer]
CONFIDENCE: [0.0-1.0]
"""
    
    def create_initial_state(self, query: str, context: Optional[Dict[str, Any]] = None) -> AgentState:
        """Create initial agent state."""
        return AgentState(
            query=query,
            thoughts=[],
            current_action=None,
            context=context or {},
            max_steps=self.max_steps,
            current_step=0,
            completed=False,
            final_answer=None,
            confidence=0.0
        )
    
    def reason(self, state: AgentState) -> Thought:
        """Generate a reasoning step."""
        try:
            # Build context for the prompt
            thoughts_text = self._format_thoughts(state.thoughts)
            context_text = self._format_context(state.context)
            
            prompt = self.react_prompt.format(
                query=state.query,
                context=context_text,
                thoughts=thoughts_text,
                current_step=state.current_step + 1,
                max_steps=state.max_steps
            )
            
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are a ReAct agent. Think step by step."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=400
            )
            
            react_output = response.choices[0].message.content.strip()
            
            # Parse the ReAct output
            thought_content = ""
            action = None
            confidence = 0.5
            answer = None
            
            for line in react_output.split('\n'):
                line = line.strip()
                if line.startswith('THOUGHT:'):
                    thought_content = line.split(':', 1)[1].strip()
                elif line.startswith('ACTION:'):
                    action_str = line.split(':', 1)[1].strip()
                    try:
                        action = AgentAction(action_str.lower())
                    except ValueError:
                        logger.warning(f"Unknown action: {action_str}")
                        action = AgentAction.REASON
                elif line.startswith('CONFIDENCE:'):
                    try:
                        confidence = float(line.split(':', 1)[1].strip())
                    except ValueError:
                        pass
                elif line.startswith('ANSWER:'):
                    answer = line.split(':', 1)[1].strip()
            
            # Create thought
            thought = Thought(
                step=state.current_step + 1,
                content=thought_content,
                action=action or AgentAction.REASON,
                observation="",  # Will be filled after action execution
                confidence=confidence,
                timestamp=datetime.now()
            )
            
            logger.info(f"ReAct step {state.current_step + 1}: {action} (confidence: {confidence})")
            return thought
            
        except Exception as e:
            logger.error(f"Error in ReAct reasoning: {e}")
            # Fallback thought
            return Thought(
                step=state.current_step + 1,
                content=f"Error during reasoning: {str(e)}",
                action=AgentAction.REASON,
                observation="",
                confidence=0.1,
                timestamp=datetime.now()
            )
    
    def execute_action(self, state: AgentState, thought: Thought) -> str:
        """Execute the chosen action and return observation."""
        try:
            if thought.action == AgentAction.REASON:
                return self._execute_reason(state, thought)
            elif thought.action == AgentAction.SEARCH_PORTFOLIO:
                return self._execute_search_portfolio(state, thought)
            elif thought.action == AgentAction.SEARCH_WEB:
                return self._execute_search_web(state, thought)
            elif thought.action == AgentAction.ANALYZE_DATA:
                return self._execute_analyze_data(state, thought)
            elif thought.action == AgentAction.GENERATE_RESPONSE:
                return self._execute_generate_response(state, thought)
            elif thought.action == AgentAction.CLARIFY:
                return self._execute_clarify(state, thought)
            elif thought.action == AgentAction.ESCALATE:
                return self._execute_escalate(state, thought)
            else:
                return f"Unknown action: {thought.action}"
                
        except Exception as e:
            logger.error(f"Error executing action {thought.action}: {e}")
            return f"Error executing action: {str(e)}"
    
    def _execute_reason(self, state: AgentState, thought: Thought) -> str:
        """Execute reasoning action."""
        return "Reasoning completed. Ready for next action."
    
    def _execute_search_portfolio(self, state: AgentState, thought: Thought) -> str:
        """Execute portfolio search action."""
        # This will be implemented with actual portfolio search
        return "Portfolio search would be executed here using vector store."
    
    def _execute_search_web(self, state: AgentState, thought: Thought) -> str:
        """Execute web search action."""
        # This would integrate with web search tools
        return "Web search would be executed here for current information."
    
    def _execute_analyze_data(self, state: AgentState, thought: Thought) -> str:
        """Execute data analysis action."""
        # This would analyze provided data or context
        return "Data analysis completed. Key insights identified."
    
    def _execute_generate_response(self, state: AgentState, thought: Thought) -> str:
        """Execute response generation action."""
        # Generate final answer based on accumulated knowledge
        try:
            final_prompt = f"""
Based on the ReAct process for query: "{state.query}"

Previous thoughts and observations:
{self._format_thoughts(state.thoughts)}

Generate a comprehensive, accurate answer to the user's query.
Be specific and provide relevant details.
"""
            
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are generating a final answer based on reasoning process."},
                    {"role": "user", "content": final_prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            final_answer = response.choices[0].message.content.strip()
            state.final_answer = final_answer
            state.completed = True
            state.confidence = thought.confidence
            
            return f"Final answer generated with confidence {thought.confidence}"
            
        except Exception as e:
            logger.error(f"Error generating final response: {e}")
            return f"Error generating response: {str(e)}"
    
    def _execute_clarify(self, state: AgentState, thought: Thought) -> str:
        """Execute clarification action."""
        return "I need more information to answer your question. Could you please clarify what you're looking for?"
    
    def _execute_escalate(self, state: AgentState, thought: Thought) -> str:
        """Execute escalation action."""
        state.completed = True
        state.final_answer = "I'm unable to answer this question with the available information and tools."
        state.confidence = 0.0
        return "Question escalated - cannot be answered with current capabilities."
    
    def _format_thoughts(self, thoughts: List[Thought]) -> str:
        """Format thoughts for display."""
        if not thoughts:
            return "No previous thoughts."
        
        formatted = []
        for thought in thoughts:
            formatted.append(f"Step {thought.step}: {thought.action.value}")
            formatted.append(f"Thought: {thought.content}")
            formatted.append(f"Observation: {thought.observation}")
            formatted.append(f"Confidence: {thought.confidence}")
            formatted.append("")
        
        return "\n".join(formatted)
    
    def _format_context(self, context: Dict[str, Any]) -> str:
        """Format context for display."""
        if not context:
            return "No additional context."
        
        formatted = []
        for key, value in context.items():
            formatted.append(f"{key}: {value}")
        
        return "\n".join(formatted)
    
    def run(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run the ReAct agent on a query.
        
        Args:
            query: User's question
            context: Optional context information
            
        Returns:
            Dictionary with agent results
        """
        logger.info(f"Starting ReAct agent for query: {query[:50]}...")
        
        # Initialize state
        state = self.create_initial_state(query, context)
        
        try:
            # Run ReAct loop
            while not state.completed and state.current_step < state.max_steps:
                state.current_step += 1
                
                # Generate thought
                thought = self.reason(state)
                
                # Execute action
                observation = self.execute_action(state, thought)
                thought.observation = observation
                
                # Add thought to state
                state.thoughts.append(thought)
                
                logger.info(f"Step {state.current_step} completed: {thought.action.value}")
                
                # Check if we should stop
                if thought.action == AgentAction.GENERATE_RESPONSE and state.final_answer:
                    break
                elif thought.action == AgentAction.ESCALATE:
                    break
            
            # Prepare results
            results = {
                "query": query,
                "completed": state.completed,
                "final_answer": state.final_answer,
                "confidence": state.confidence,
                "steps_taken": state.current_step,
                "thoughts": [
                    {
                        "step": t.step,
                        "action": t.action.value,
                        "thought": t.content,
                        "observation": t.observation,
                        "confidence": t.confidence
                    }
                    for t in state.thoughts
                ],
                "context_used": context or {}
            }
            
            logger.info(f"ReAct agent completed in {state.current_step} steps with confidence {state.confidence}")
            return results
            
        except Exception as e:
            logger.error(f"Error in ReAct agent: {e}")
            return {
                "query": query,
                "completed": False,
                "final_answer": "An error occurred during processing.",
                "confidence": 0.0,
                "steps_taken": state.current_step,
                "error": str(e),
                "thoughts": []
            }