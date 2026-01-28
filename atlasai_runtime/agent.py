"""
Simple agent implementation using LangGraph for orchestrating RAG and task execution.
This demonstrates how to extend AtlasAI with agent-based workflows.
"""

import logging
from typing import Dict, Any, List, Optional, TypedDict
from enum import Enum

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """State for the agent workflow."""
    query: str
    intent: Optional[str]
    needs_docs: bool
    needs_task: bool
    doc_results: Optional[Dict[str, Any]]
    task_results: Optional[Dict[str, Any]]
    final_answer: Optional[str]
    error: Optional[str]


class IntentType(Enum):
    """Intent types for routing."""
    QUESTION = "question"
    TASK = "task"
    BOTH = "both"


class SimpleAgent:
    """
    A simple agent that can route between RAG queries and task execution.
    This is a foundation for more complex LangGraph-based workflows.
    """
    
    def __init__(self, rag_engine, task_executor):
        """
        Initialize the agent.
        
        Args:
            rag_engine: RAG engine instance for document queries
            task_executor: Task executor instance for system commands
        """
        self.rag_engine = rag_engine
        self.task_executor = task_executor
    
    def classify_intent(self, query: str) -> IntentType:
        """
        Classify whether the query needs RAG, task execution, or both.
        
        Args:
            query: User query
            
        Returns:
            IntentType indicating the workflow needed
        """
        query_lower = query.lower()
        
        # Keywords that suggest task execution
        task_keywords = [
            "run", "execute", "open", "list files", "show directory",
            "check system", "get info", "command"
        ]
        
        # Check if query contains task keywords
        has_task_intent = any(keyword in query_lower for keyword in task_keywords)
        
        # Keywords that suggest document query
        doc_keywords = [
            "what is", "how to", "explain", "describe", "tell me about",
            "documentation", "manual", "guide", "instructions"
        ]
        
        has_doc_intent = any(keyword in query_lower for keyword in doc_keywords)
        
        # Route based on intent
        if has_task_intent and has_doc_intent:
            return IntentType.BOTH
        elif has_task_intent:
            return IntentType.TASK
        else:
            return IntentType.QUESTION
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a query using the agent workflow.
        
        Args:
            query: User query
            
        Returns:
            Dictionary with results
        """
        # Initialize state
        state: AgentState = {
            "query": query,
            "intent": None,
            "needs_docs": False,
            "needs_task": False,
            "doc_results": None,
            "task_results": None,
            "final_answer": None,
            "error": None
        }
        
        try:
            # Step 1: Classify intent
            intent = self.classify_intent(query)
            state["intent"] = intent.value
            
            # Step 2: Determine what's needed
            state["needs_docs"] = intent in [IntentType.QUESTION, IntentType.BOTH]
            state["needs_task"] = intent in [IntentType.TASK, IntentType.BOTH]
            
            # Step 3: Execute RAG query if needed
            if state["needs_docs"]:
                logger.info("Querying RAG engine...")
                state["doc_results"] = self.rag_engine.query(query)
            
            # Step 4: Execute task if needed
            if state["needs_task"]:
                # Extract command from query (simple version)
                command = self._extract_command(query)
                if command:
                    logger.info(f"Executing command: {command}")
                    state["task_results"] = self.task_executor.execute_command(command)
            
            # Step 5: Combine results
            state["final_answer"] = self._combine_results(state)
            
        except Exception as e:
            logger.error(f"Error in agent workflow: {e}")
            state["error"] = str(e)
        
        return state
    
    def _extract_command(self, query: str) -> Optional[str]:
        """
        Extract a command from the query.
        This is a simple implementation; a real version would be more sophisticated.
        
        Args:
            query: User query
            
        Returns:
            Extracted command or None
        """
        query_lower = query.lower()
        
        # Simple patterns
        if "python version" in query_lower or "check python" in query_lower:
            return "python --version"
        elif "list files" in query_lower or "show files" in query_lower:
            return "ls -la"
        elif "current directory" in query_lower or "where am i" in query_lower:
            return "pwd"
        elif "system info" in query_lower:
            # Return None to use task_executor.get_system_info() instead
            return None
        
        # If query starts with "run" or "execute", try to extract command
        if query_lower.startswith("run "):
            return query[4:].strip()
        elif query_lower.startswith("execute "):
            return query[8:].strip()
        
        return None
    
    def _combine_results(self, state: AgentState) -> str:
        """
        Combine RAG and task results into a final answer.
        
        Args:
            state: Current agent state
            
        Returns:
            Combined answer string
        """
        parts = []
        
        # Add document results
        if state["doc_results"]:
            parts.append(state["doc_results"]["answer"])
            if state["doc_results"]["sources"]:
                parts.append("\nSources:")
                for src in state["doc_results"]["sources"]:
                    parts.append(f"  {src['index']}. {src['source']} (page {src['page']})")
        
        # Add task results
        if state["task_results"]:
            if state["task_results"]["success"]:
                parts.append("\nTask execution result:")
                parts.append(state["task_results"]["stdout"])
            else:
                parts.append("\nTask execution failed:")
                parts.append(state["task_results"].get("error", "Unknown error"))
        
        # Combine
        if parts:
            return "\n".join(parts)
        else:
            return "I couldn't process your request. Please try rephrasing."


# Example usage and workflow definition
def create_agent_workflow(rag_engine, task_executor):
    """
    Create a simple agent workflow.
    
    This is a foundation that can be extended with LangGraph for more
    complex multi-step workflows, branching logic, and state management.
    
    Args:
        rag_engine: RAG engine instance
        task_executor: Task executor instance
        
    Returns:
        SimpleAgent instance
    """
    return SimpleAgent(rag_engine, task_executor)
