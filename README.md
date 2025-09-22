# 🧠 Deep Researcher Agent

An AI-powered multi-document research assistant that runs **locally** using [Ollama](https://ollama.ai).  
This guide will help you set up and run the project on your machine.

---

## 🚀 Setup Instructions

### Step 1: Clone Repository
```bash
git clone https://github.com/mohithpranav/deep_research_agent
cd deep_researcher_agent

# Create virtual environment
python -m venv venv

# Activate environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Upgrade pip
python -m pip install --upgrade pip

# Install all required packages
pip install -r requirements.txt

# Install any missing packages
pip install markdown beautifulsoup4 lxml typing-extensions

# Windows (via winget or download installer)
winget install Ollama.Ollama

# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama service (keep this terminal running)
ollama serve

# Open NEW terminal and pull the model (4GB download)
ollama pull llama2:7b

# Test the model works
ollama run llama2:7b "Hello, introduce yourself briefly"

# Navigate back to project directory
cd deep_researcher_agent

# Test Ollama connection and llama2:7b model
python test_ollama.py

# Test all AI components
python debug_ai.py

# Clean any existing database
python cleanup_database.py

# Start the Deep Researcher Agent
streamlit run main.py

