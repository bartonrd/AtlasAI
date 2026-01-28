"""
Agent System with LangGraph for local task execution.
Enables the chatbot to perform actions on the user's machine.
"""

import os
import subprocess
import platform
import logging
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor, ToolInvocation
from langchain_core.tools import tool

logger = logging.getLogger(__name__)


# Define the agent state
class AgentState(TypedDict):
    """State of the agent."""
    messages: Annotated[Sequence[BaseMessage], "The messages in the conversation"]
    task_result: str
    error: str


# Define tools for local task execution
@tool
def execute_shell_command(command: str) -> str:
    """
    Execute a shell command on the local machine.
    Use with caution - only for safe, non-destructive operations.
    
    Args:
        command: The shell command to execute
        
    Returns:
        The output of the command or error message
    """
    try:
        # Security check - prevent dangerous commands
        dangerous_keywords = ['rm -rf', 'format', 'del /f', 'rmdir /s', 'mkfs', 'dd if=']
        command_lower = command.lower()
        
        for keyword in dangerous_keywords:
            if keyword in command_lower:
                return f"ERROR: Command contains dangerous keyword '{keyword}' and was blocked for safety."
        
        # Execute command with timeout
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        output = result.stdout if result.stdout else result.stderr
        return f"Exit code: {result.returncode}\nOutput:\n{output}"
        
    except subprocess.TimeoutExpired:
        return "ERROR: Command timed out after 30 seconds"
    except Exception as e:
        return f"ERROR: Failed to execute command: {str(e)}"


@tool
def list_directory(path: str = ".") -> str:
    """
    List files and directories at the specified path.
    
    Args:
        path: The directory path to list (default: current directory)
        
    Returns:
        List of files and directories
    """
    try:
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            return f"ERROR: Path does not exist: {path}"
        
        if not os.path.isdir(path):
            return f"ERROR: Path is not a directory: {path}"
        
        items = os.listdir(path)
        result = f"Contents of {path}:\n"
        
        for item in sorted(items):
            item_path = os.path.join(path, item)
            if os.path.isdir(item_path):
                result += f"  [DIR]  {item}\n"
            else:
                size = os.path.getsize(item_path)
                result += f"  [FILE] {item} ({size} bytes)\n"
        
        return result
        
    except Exception as e:
        return f"ERROR: Failed to list directory: {str(e)}"


@tool
def read_file(path: str, max_lines: int = 100) -> str:
    """
    Read the contents of a text file.
    
    Args:
        path: The file path to read
        max_lines: Maximum number of lines to read (default: 100)
        
    Returns:
        The file contents or error message
    """
    try:
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            return f"ERROR: File does not exist: {path}"
        
        if not os.path.isfile(path):
            return f"ERROR: Path is not a file: {path}"
        
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = []
            for i, line in enumerate(f):
                if i >= max_lines:
                    lines.append(f"\n... (truncated after {max_lines} lines)")
                    break
                lines.append(line.rstrip())
            
            return '\n'.join(lines)
        
    except Exception as e:
        return f"ERROR: Failed to read file: {str(e)}"


@tool
def get_system_info() -> str:
    """
    Get information about the system.
    
    Returns:
        System information including OS, platform, and Python version
    """
    try:
        import sys
        
        info = f"""System Information:
- Operating System: {platform.system()} {platform.release()}
- Platform: {platform.platform()}
- Machine: {platform.machine()}
- Processor: {platform.processor()}
- Python Version: {sys.version}
- Current Directory: {os.getcwd()}
- Home Directory: {os.path.expanduser('~')}
"""
        return info
        
    except Exception as e:
        return f"ERROR: Failed to get system info: {str(e)}"


@tool
def write_file(path: str, content: str) -> str:
    """
    Write content to a file.
    
    Args:
        path: The file path to write to
        content: The content to write
        
    Returns:
        Success or error message
    """
    try:
        path = os.path.expanduser(path)
        
        # Create directory if it doesn't exist
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return f"Successfully wrote {len(content)} characters to {path}"
        
    except Exception as e:
        return f"ERROR: Failed to write file: {str(e)}"


# Available tools
AVAILABLE_TOOLS = [
    execute_shell_command,
    list_directory,
    read_file,
    get_system_info,
    write_file,
]


class LocalTaskAgent:
    """
    Agent for executing local tasks using LangGraph.
    """
    
    def __init__(self, ollama_model: str = "llama3.1:8b-instruct-q4_0", 
                 ollama_base_url: str = "http://localhost:11434"):
        """
        Initialize the agent.
        
        Args:
            ollama_model: The Ollama model to use
            ollama_base_url: The Ollama server URL
        """
        self.ollama_model = ollama_model
        self.ollama_base_url = ollama_base_url
        
        # Initialize LLM
        self.llm = ChatOllama(
            model=ollama_model,
            base_url=ollama_base_url,
            temperature=0.1,
        )
        
        # Bind tools to LLM
        self.llm_with_tools = self.llm.bind_tools(AVAILABLE_TOOLS)
        
        # Create tool executor
        self.tool_executor = ToolExecutor(AVAILABLE_TOOLS)
        
        # Build the graph
        self.graph = self._build_graph()
        
        logger.info("LocalTaskAgent initialized with LangGraph")
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        
        # Define the agent node
        def call_model(state: AgentState):
            messages = state["messages"]
            response = self.llm_with_tools.invoke(messages)
            return {"messages": messages + [response]}
        
        # Define the tool execution node
        def call_tool(state: AgentState):
            messages = state["messages"]
            last_message = messages[-1]
            
            # Execute tool calls
            tool_calls = last_message.tool_calls if hasattr(last_message, 'tool_calls') else []
            
            if not tool_calls:
                return {"messages": messages}
            
            # Execute each tool call
            for tool_call in tool_calls:
                try:
                    tool_invocation = ToolInvocation(
                        tool=tool_call["name"],
                        tool_input=tool_call["args"],
                    )
                    result = self.tool_executor.invoke(tool_invocation)
                    
                    # Add tool result as a message
                    tool_message = AIMessage(content=str(result))
                    messages = messages + [tool_message]
                    
                except Exception as e:
                    error_message = AIMessage(content=f"ERROR executing tool: {str(e)}")
                    messages = messages + [error_message]
            
            return {"messages": messages}
        
        # Define routing logic
        def should_continue(state: AgentState):
            messages = state["messages"]
            last_message = messages[-1]
            
            # If there are tool calls, continue to tool execution
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                return "continue"
            
            # Otherwise, end
            return "end"
        
        # Create the graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("agent", call_model)
        workflow.add_node("tool", call_tool)
        
        # Set entry point
        workflow.set_entry_point("agent")
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "agent",
            should_continue,
            {
                "continue": "tool",
                "end": END,
            }
        )
        
        # Add edge from tool back to agent
        workflow.add_edge("tool", "agent")
        
        # Compile the graph
        return workflow.compile()
    
    def execute_task(self, task_description: str) -> str:
        """
        Execute a task described in natural language.
        
        Args:
            task_description: Natural language description of the task
            
        Returns:
            The result of the task execution
        """
        try:
            # Create initial state
            system_prompt = """You are a helpful AI assistant that can execute tasks on the user's local machine.
You have access to tools for:
- Executing shell commands (use with caution)
- Listing directories
- Reading files
- Getting system information
- Writing files

When the user asks you to do something, use the appropriate tools to accomplish the task.
Be careful with destructive operations and always confirm what you're doing.
Provide clear, concise responses about what you did and the results.
"""
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=task_description),
            ]
            
            initial_state = AgentState(
                messages=messages,
                task_result="",
                error=""
            )
            
            # Execute the graph
            logger.info(f"Executing task: {task_description}")
            final_state = self.graph.invoke(initial_state)
            
            # Extract the final response
            final_messages = final_state["messages"]
            
            # Get the last AI message
            for message in reversed(final_messages):
                if isinstance(message, AIMessage) and message.content:
                    return message.content
            
            return "Task completed, but no response was generated."
            
        except Exception as e:
            logger.error(f"Error executing task: {e}", exc_info=True)
            return f"ERROR: Failed to execute task: {str(e)}"
    
    def can_handle_task(self, query: str) -> bool:
        """
        Determine if a query is a task that should be handled by the agent.
        
        Args:
            query: The user's query
            
        Returns:
            True if the agent should handle this task
        """
        # Keywords that indicate a local task
        task_keywords = [
            'run', 'execute', 'create file', 'write file', 'list', 'show files',
            'directory', 'folder', 'read file', 'system info', 'check system',
            'command', 'script', 'open', 'launch', 'start'
        ]
        
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in task_keywords)
