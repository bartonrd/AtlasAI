"""
AtlasAI - Improved Modular RAG Chatbot Application
Main entry point for the Streamlit application
"""

from src.config import Settings
from src.services import RAGService
from src.ui import ChatInterface


def main():
    """Main application entry point"""
    # Load settings
    settings = Settings.load_from_env()
    
    # Initialize RAG service
    rag_service = RAGService(settings)
    
    # Create and render chat interface
    chat_interface = ChatInterface(settings, rag_service)
    chat_interface.render()


if __name__ == "__main__":
    main()
