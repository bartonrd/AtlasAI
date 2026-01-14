"""
Validation script for AtlasAI architecture
Tests that all modules can be imported and basic functionality works
"""

import sys
from pathlib import Path

def test_imports():
    """Test that all modules can be imported"""
    print("Testing module imports...")
    
    try:
        from src.config import Settings
        print("✓ Config module imported")
    except ImportError as e:
        print(f"✗ Failed to import config module: {e}")
        return False
    
    try:
        from src.services import DocumentService, EmbeddingService, LLMService, RAGService
        print("✓ Service modules imported")
    except ImportError as e:
        print(f"✗ Failed to import service modules: {e}")
        return False
    
    try:
        from src.utils import format_answer_as_bullets, thinking_message
        print("✓ Utility modules imported")
    except ImportError as e:
        print(f"✗ Failed to import utility modules: {e}")
        return False
    
    try:
        from src.ui import ChatInterface
        print("✓ UI module imported")
    except ImportError as e:
        print(f"✗ Failed to import UI module: {e}")
        return False
    
    return True

def test_config():
    """Test configuration management"""
    print("\nTesting configuration...")
    
    from src.config import Settings
    
    try:
        settings = Settings()
        print(f"✓ Settings created with defaults")
        print(f"  - Embedding model: {settings.model.embedding_model}")
        print(f"  - Text gen model: {settings.model.text_gen_model}")
        print(f"  - Chunk size: {settings.rag.chunk_size}")
        print(f"  - Top K: {settings.rag.top_k}")
        
        # Test validation
        errors = settings.rag.validate()
        if not errors:
            print("✓ Settings validation passed")
        else:
            print(f"✗ Settings validation failed: {errors}")
            return False
        
        return True
    except Exception as e:
        print(f"✗ Config test failed: {e}")
        return False

def test_document_service():
    """Test document service"""
    print("\nTesting document service...")
    
    from src.services import DocumentService
    
    try:
        service = DocumentService()
        print("✓ DocumentService created")
        
        # Test boilerplate removal
        text = "Some text\nProprietary - See Copyright Page\nMore text"
        cleaned = service.strip_boilerplate(text)
        if "Proprietary" not in cleaned:
            print("✓ Boilerplate removal works")
        else:
            print("✗ Boilerplate removal failed")
            return False
        
        return True
    except Exception as e:
        print(f"✗ DocumentService test failed: {e}")
        return False

def test_formatting():
    """Test formatting utilities"""
    print("\nTesting formatting utilities...")
    
    from src.utils import format_answer_as_bullets, thinking_message
    
    try:
        # Test bullet formatting
        text = "First point. Second point. Third point."
        bullets = format_answer_as_bullets(text)
        if bullets.startswith("- "):
            print("✓ Bullet formatting works")
        else:
            print("✗ Bullet formatting failed")
            return False
        
        # Test thinking message
        msg = thinking_message("Processing...")
        if "Processing..." in msg and "style" in msg:
            print("✓ Thinking message formatting works")
        else:
            print("✗ Thinking message formatting failed")
            return False
        
        return True
    except Exception as e:
        print(f"✗ Formatting test failed: {e}")
        return False

def main():
    """Run all validation tests"""
    print("=" * 60)
    print("AtlasAI Architecture Validation")
    print("=" * 60)
    
    tests = [
        ("Module Imports", test_imports),
        ("Configuration", test_config),
        ("Document Service", test_document_service),
        ("Formatting Utilities", test_formatting),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} test crashed: {e}")
            results.append((name, False))
    
    # Print summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    print("=" * 60)
    print(f"Results: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("\n🎉 All validation tests passed!")
        print("The modular architecture is working correctly.")
        return 0
    else:
        print("\n⚠️  Some validation tests failed.")
        print("Please review the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
