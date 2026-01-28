"""
Local Task Executor - Execute system commands and tasks on the user's machine.
"""

import os
import subprocess
import platform
import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


class LocalTaskExecutor:
    """
    Executes local system tasks based on user commands.
    """

    def __init__(self, allowed_commands: Optional[List[str]] = None):
        """
        Initialize the task executor.
        
        Args:
            allowed_commands: List of allowed command prefixes for security
        """
        # Default safe commands
        if allowed_commands is None:
            allowed_commands = [
                "ls", "dir", "pwd", "cd", "cat", "type", "echo",
                "python", "pip", "git", "dotnet", "npm", "node",
                "code", "notepad", "vim", "nano",
                "mkdir", "touch", "cp", "mv", "rm"
            ]
        self.allowed_commands = set(allowed_commands)
        self.system = platform.system()

    def is_command_allowed(self, command: str) -> bool:
        """
        Check if a command is allowed to be executed.
        
        Args:
            command: Command string to check
            
        Returns:
            True if command is allowed
        """
        # Get the first word of the command
        cmd_parts = command.strip().split()
        if not cmd_parts:
            return False
        
        cmd_name = cmd_parts[0].lower()
        
        # Check against allowed list
        return cmd_name in self.allowed_commands

    def execute_command(
        self,
        command: str,
        working_dir: Optional[str] = None,
        timeout: int = 30
    ) -> Dict[str, Any]:
        """
        Execute a system command.
        
        Args:
            command: Command to execute
            working_dir: Working directory for command execution
            timeout: Timeout in seconds
            
        Returns:
            Dictionary with execution results
        """
        logger.info(f"Executing command: {command}")
        
        # Security check
        if not self.is_command_allowed(command):
            return {
                "success": False,
                "error": f"Command not allowed: {command.split()[0]}",
                "stdout": "",
                "stderr": "",
                "return_code": -1
            }
        
        try:
            # Set working directory
            cwd = working_dir or os.getcwd()
            
            # Execute command
            result = subprocess.run(
                command,
                shell=True,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode,
                "error": None
            }
            
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": f"Command timed out after {timeout} seconds",
                "stdout": "",
                "stderr": "",
                "return_code": -1
            }
        except Exception as e:
            logger.error(f"Error executing command: {e}")
            return {
                "success": False,
                "error": str(e),
                "stdout": "",
                "stderr": "",
                "return_code": -1
            }

    def get_system_info(self) -> Dict[str, str]:
        """
        Get system information.
        
        Returns:
            Dictionary with system information
        """
        return {
            "system": platform.system(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "cwd": os.getcwd()
        }

    def list_directory(self, path: str = ".") -> Dict[str, Any]:
        """
        List contents of a directory.
        
        Args:
            path: Directory path
            
        Returns:
            Dictionary with directory contents
        """
        try:
            path = os.path.abspath(path)
            
            if not os.path.exists(path):
                return {
                    "success": False,
                    "error": f"Path does not exist: {path}",
                    "files": [],
                    "directories": []
                }
            
            if not os.path.isdir(path):
                return {
                    "success": False,
                    "error": f"Path is not a directory: {path}",
                    "files": [],
                    "directories": []
                }
            
            items = os.listdir(path)
            files = []
            directories = []
            
            for item in items:
                item_path = os.path.join(path, item)
                if os.path.isfile(item_path):
                    files.append({
                        "name": item,
                        "size": os.path.getsize(item_path)
                    })
                elif os.path.isdir(item_path):
                    directories.append(item)
            
            return {
                "success": True,
                "path": path,
                "files": files,
                "directories": directories,
                "error": None
            }
            
        except Exception as e:
            logger.error(f"Error listing directory: {e}")
            return {
                "success": False,
                "error": str(e),
                "files": [],
                "directories": []
            }

    def read_file(self, path: str, max_size: int = 1024 * 1024) -> Dict[str, Any]:
        """
        Read contents of a text file.
        
        Args:
            path: File path
            max_size: Maximum file size to read (default 1MB)
            
        Returns:
            Dictionary with file contents
        """
        try:
            path = os.path.abspath(path)
            
            if not os.path.exists(path):
                return {
                    "success": False,
                    "error": f"File does not exist: {path}",
                    "content": ""
                }
            
            if not os.path.isfile(path):
                return {
                    "success": False,
                    "error": f"Path is not a file: {path}",
                    "content": ""
                }
            
            file_size = os.path.getsize(path)
            if file_size > max_size:
                return {
                    "success": False,
                    "error": f"File too large: {file_size} bytes (max: {max_size} bytes)",
                    "content": ""
                }
            
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            return {
                "success": True,
                "path": path,
                "content": content,
                "size": file_size,
                "error": None
            }
            
        except UnicodeDecodeError:
            return {
                "success": False,
                "error": "File is not a text file or has unsupported encoding",
                "content": ""
            }
        except Exception as e:
            logger.error(f"Error reading file: {e}")
            return {
                "success": False,
                "error": str(e),
                "content": ""
            }

    def open_file(self, path: str) -> Dict[str, Any]:
        """
        Open a file with the default system application.
        
        Args:
            path: File path
            
        Returns:
            Dictionary with operation result
        """
        try:
            path = os.path.abspath(path)
            
            if not os.path.exists(path):
                return {
                    "success": False,
                    "error": f"File does not exist: {path}"
                }
            
            # Platform-specific open commands
            if self.system == "Windows":
                os.startfile(path)
            elif self.system == "Darwin":  # macOS
                subprocess.run(["open", path], check=True)
            else:  # Linux
                subprocess.run(["xdg-open", path], check=True)
            
            return {
                "success": True,
                "path": path,
                "error": None
            }
            
        except Exception as e:
            logger.error(f"Error opening file: {e}")
            return {
                "success": False,
                "error": str(e)
            }
