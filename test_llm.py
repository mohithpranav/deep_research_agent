"""
Quick LLM Test Script
"""
import requests
import json

def test_ollama():
    """Test if Ollama is working"""
    try:
        # Check models
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        print(f"Response headers: {dict(response.headers)}")
        if response.status_code == 200:
            models = response.json().get('models', [])
            print(f"✅ Ollama is running with {len(models)} models:")
            for model in models:
                print(f"  - {model['name']}")
            
            if models:
                # Test a query
                model_name = models[0]['name']
                test_query = "Explain what artificial intelligence is in one sentence."
                
                payload = {
                    "model": model_name,
                    "prompt": test_query,
                    "stream": False
                }
                
                print(f"\n🧪 Testing with query: '{test_query}'")
                response = requests.post("http://localhost:11434/api/generate", json=payload, timeout=30)
                
                if response.status_code == 200:
                    result = response.json()
                    llm_response = result.get('response', '')
                    print(f"🤖 LLM Response: {llm_response}")
                    print(f"✅ LLM is working correctly!")
                else:
                    print(f"❌ LLM generate failed: {response.status_code}")
            else:
                print("❌ No models available. Run: ollama pull llama2:7b")
        else:
            print(f"❌ Ollama not responding: {response.status_code}")
    
    except Exception as e:
        print(f"❌ Ollama connection failed: {e}")
        print("🔧 Make sure Ollama is running: ollama serve")

def test_frontend_connection():
    """Test connection as if from frontend"""
    headers = {
        'Content-Type': 'application/json',
        'Origin': 'http://localhost:3000',  # Adjust to your frontend port
        'Access-Control-Request-Method': 'POST'
    }
    
    try:
        response = requests.options("http://localhost:11434/api/generate", headers=headers)
        print(f"CORS preflight: {response.status_code}")
        print(f"CORS headers: {dict(response.headers)}")
    except Exception as e:
        print(f"Frontend connection test failed: {e}")

if __name__ == "__main__":
    test_ollama()
    test_frontend_connection()