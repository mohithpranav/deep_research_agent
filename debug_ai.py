"""
Debug AI connection issues
"""
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

def debug_ai_system():
    print("🔍 DEBUGGING AI SYSTEM")
    print("=" * 50)
    
    # Test 1: Basic imports
    try:
        from reasoning.llm_client import LLMClient
        from reasoning.query_analyzer import QueryAnalyzer
        from reasoning.response_synthesizer import ResponseSynthesizer
        print("✅ All reasoning modules imported successfully")
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False
    
    # Test 2: LLM Client health
    try:
        llm_client = LLMClient()
        is_healthy = llm_client.is_healthy()
        print(f"🏥 LLM Client Health: {'✅ Healthy' if is_healthy else '❌ Issues'}")
        
        if not is_healthy:
            print("🔧 Trying to diagnose Ollama connection...")
            import requests
            try:
                response = requests.get("http://localhost:11434/api/tags", timeout=5)
                print(f"📡 Ollama API Response: {response.status_code}")
                if response.status_code == 200:
                    models = response.json()
                    print(f"📦 Available models: {[m['name'] for m in models.get('models', [])]}")
            except Exception as api_error:
                print(f"❌ Ollama API Error: {api_error}")
                
    except Exception as e:
        print(f"❌ LLM Client error: {e}")
        return False
    
    # Test 3: Full reasoning pipeline
    try:
        query_analyzer = QueryAnalyzer()
        synthesizer = ResponseSynthesizer()
        
        test_query = "Who is Mohith?"
        test_context = ["Mohith Pranav is a Full-Stack Developer"]
        
        print(f"\n🧪 Testing reasoning pipeline with: '{test_query}'")
        
        # Analyze query
        query_info = query_analyzer.analyze(test_query)
        print(f"✅ Query Analysis: {query_info['query_type']}")
        
        # Test synthesis
        if synthesizer.is_healthy():
            print("🧠 Testing AI synthesis...")
            response = synthesizer.synthesize(test_query, test_context, query_info)
            print(f"✅ Synthesis result: {len(response)} characters")
            print(f"📝 Preview: {response[:100]}...")
        else:
            print("❌ Synthesizer not healthy")
            
    except Exception as e:
        print(f"❌ Pipeline error: {e}")
        return False
    
    print("\n🎉 AI system debugging complete!")
    return True

if __name__ == "__main__":
    success = debug_ai_system()
    if not success:
        print("\n❌ Please fix the issues above before running the main app")
    else:
        print("✅ AI system should work properly now!")