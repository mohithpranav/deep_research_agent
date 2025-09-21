3️⃣ **Clone Repo & Setup**

```bash
git clone https://github.com/mohithpranav/deep_research_agent
cd deep_researcher_agent
python -m venv venv

# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# Install all dependencies
pip install -r requirements.txt

# Install any missing dependencies
pip install markdown beautifulsoup4 lxml

# Test setup
python test_ollama.py

# Run application
streamlit run main.py
```
