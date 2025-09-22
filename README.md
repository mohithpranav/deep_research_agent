Step 1: Clone Repository

git clone https://github.com/mohithpranav/deep_research_agent
cd deep_researcher_agent

Step 2: Python Environment Setup

# Create virtual environment

python -m venv venv

# Activate environment

# Windows:

venv\Scripts\activate

# macOS/Linux:

source venv/bin/activate

# Upgrade pip

python -m pip install --upgrade pip

Step 3: Install Python Dependencies

# Install all required packages

pip install -r requirements.txt

# Install any missing packages

pip install markdown beautifulsoup4 lxml typing-extensions

Step 4: Install Ollama (Local AI Engine)

# Windows - Download installer from ollama.ai

# Or use winget:

winget install Ollama.Ollama

# macOS:

brew install ollama

# Linux:

curl -fsSL https://ollama.ai/install.sh | sh

Step 5: Setup llama2:7b Model

# Start Ollama service (keep this terminal running)

ollama serve

# Open NEW terminal and pull the model (4GB download):

ollama pull llama2:7b

# Test the model works:

ollama run llama2:7b "Hello, introduce yourself briefly"

Step 6: Verify Complete Setup

# Navigate back to project directory

cd deep_researcher_agent

# Test Ollama connection and llama2:7b model

python test_ollama.py

# Expected output:

# ✅ Ollama server is running

# 📦 Available models: ['llama2:7b']

# ✅ llama2:7b is available

# ✅ Model response: Hello! I'm LLaMA...

# 🎉 Ollama setup is working perfectly!

# Test all AI components

python debug_ai.py

# Clean any existing database

python cleanup_database.py

Step 7: Run the Application

# Start the Deep Researcher Agent

streamlit run main.py

# App automatically opens at: http://localhost:8501

📦 Complete Requirements
Your requirements.txt includes:
streamlit>=1.28.0 # Web UI framework
sentence-transformers>=2.2.2 # Text embeddings  
pdfplumber>=0.9.0 # PDF processing
python-docx>=0.8.11 # Word document processing
faiss-cpu>=1.7.4 # Vector similarity search
pandas>=2.0.0 # Data manipulation
numpy>=1.24.0 # Numerical computing
python-dotenv>=1.0.0 # Environment variables
markdown>=3.5.1 # Markdown processing
typing-extensions>=4.8.0 # Type hints
beautifulsoup4>=4.12.0 # HTML parsing
lxml>=4.9.0 # XML processing
ollama>=0.1.7 # Ollama Python client

Ollama Issues

# Check if Ollama is running

curl http://localhost:11434/api/tags

# If not running, start it

ollama serve

# If model missing

ollama pull llama2:7b

# Test connection

python test_ollama.py

Python Issues

# If package installation fails

python -m pip install --upgrade pip
pip install -r requirements.txt --force-reinstall

# If import errors

pip install typing-extensions dataclasses

Memory Issues

# llama2:7b requires significant RAM

# Close other applications

# Or use smaller model:

ollama pull llama2:7b-q4_0 # Quantized version (less RAM)

Application Issues

# Clear database if corrupted

python cleanup_database.py

# Check logs for errors

cat logs/deep*researcher*\*.log

# Restart fresh

rm data/\*.db
streamlit run main.py
