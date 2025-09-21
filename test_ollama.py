"""
Test Ollama connection with llama2:7b
"""
import requests
import json

def test_ollama():
    """Test Ollama connection and model"""
    base_url = "http://localhost:11434"
    model = "llama2:7b"
    
    print("🧪 Testing Ollama connection...")
    
    try:
        # Test 1: Check if Ollama is running
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        if response.status_code == 200:
            print("✅ Ollama server is running")
            
            models = response.json()
            available_models = [m['name'] for m in models.get('models', [])]
            print(f"📦 Available models: {available_models}")
            
            if model in available_models:
                print(f"✅ {model} is available")
            else:
                print(f"❌ {model} not found. Please run: ollama pull {model}")
                return False
        else:
            print(f"❌ Ollama server not responding: {response.status_code}")
            return False
    
    except Exception as e:
        print(f"❌ Failed to connect to Ollama: {e}")
        print("💡 Make sure Ollama is installed and running:")
        print("   1. Download from: https://ollama.ai")
        print("   2. Run: ollama serve")
        print("   3. Run: ollama pull llama2:7b")
        return False
    
    # Test 2: Try generating a response
    print(f"\n🧠 Testing {model} generation...")
    try:
        payload = {
            "model": model,
            "prompt": "Hello! Please introduce yourself briefly.",
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_predict": 100
            }
        }
        
        response = requests.post(
            f"{base_url}/api/generate",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            generated_text = result.get('response', '')
            print(f"✅ Model response: {generated_text[:200]}...")
            return True
        else:
            print(f"❌ Generation failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Generation test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_ollama()
    if success:
        print("\n🎉 Ollama setup is working perfectly!")
        print("✅ Your Deep Researcher Agent is ready to use llama2:7b")
    else:
        print("\n❌ Please fix Ollama setup before running the main app")