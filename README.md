# 🔬 Deep Researcher Agent

> **AI-Powered Multi-Document Analysis & Research Assistant**  
> Upload multiple documents and ask complex questions across all sources. Powered by **local LLMs (Ollama)** for 100% private, offline analysis.

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/) [![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io) [![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-green.svg)](https://ollama.ai)

---

## ✨ Features

- 🔍 Multi-Document Analysis (PDF/TXT/DOCX, up to 10 files, 50MB limit)
- 🧠 Dual AI Modes – Quick Search (5s) or Deep Thinking (60s)
- 💾 Local Resource Management with SQLite storage
- 🔒 100% Offline – private on your machine
- ⚡ Real-time document indexing & retrieval
- 📊 Context-aware responses with source attribution

---

## 🖥️ Requirements

- **OS:** Windows 10/11, macOS 10.15+, or Ubuntu 18.04+
- **Python:** 3.11+
- **RAM:** 8GB (16GB recommended)
- **Storage:** 5GB free for models & documents
- **Internet:** Only for setup (model download)

---

## 📥 Installation

1️⃣ **Install Python**

- Windows: `winget install Python.Python.3.11`
- macOS: `brew install python@3.11`
- Linux: `sudo apt install python3.11 python3.11-pip python3.11-venv`

2️⃣ **Install Ollama**

- Download from [ollama.ai](https://ollama.ai) or `winget install Ollama.Ollama`
- Run service: `ollama serve`
- Pull model: `ollama pull llama2`

3️⃣ **Clone Repo & Setup**

```bash
git clone https://github.com/mohithpranav/deep_research_agent
cd deep_researcher_agent
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
pip install -r requirements.txt

# Setup
python test_ollama.py


# Run
streamlit run main.py
```

---
