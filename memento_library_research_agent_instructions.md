# 🚀 How to Run `memento_library_research_agent.py`

## 📋 Prerequisites

1. **Python 3.8+** installed
2. **Memento repository** cloned in the correct location
3. **API Keys** for LLM services
4. **Dependencies** installed

---

## 🔧 Step 1: Environment Setup

### 1.1 Verify Memento Repository
```bash
# Navigate to your project directory
cd C:\Users\PLNAYAK\Documents\Agent_Fly

# Verify Memento directory exists
dir Memento
# Should show: client/, server/, requirements.txt, README.md
```

### 1.2 Install Memento Dependencies
```bash
# Install Memento's core dependencies
pip install -r Memento/requirements.txt
```

### 1.3 Install Additional Research Agent Dependencies
```bash
# Install additional packages needed for the research agent
pip install python-dotenv asyncio requests beautifulsoup4 PyPDF2
```

---

## 🔑 Step 2: API Key Configuration

### 2.1 Create Environment File
```bash
# Create .env file in the project root
copy env_template.txt .env
```

### 2.2 Edit .env File
Open `.env` and add your API keys:
```env
# Required: Choose ONE of these
OPENAI_API_KEY=sk-your-openai-key-here
# OR
GOOGLE_API_KEY=your-google-api-key-here

# Optional: Enhanced search
SERPAPI_API_KEY=your-serpapi-key-here
GOOGLE_CSE_ID=your-google-cse-id-here

# System Configuration
LOG_LEVEL=INFO
MEMORY_BANK_PATH=memento_research_memory.json
```

---

## 🏃‍♂️ Step 3: Running the Agent

### 3.1 Basic Execution
```bash
# Navigate to the project directory
cd C:\Users\PLNAYAK\Documents\Agent_Fly

# Run the agent
python memento_library_research_agent.py
```

### 3.2 Expected Output
```
🧠 Memento Library Research Agent
================================

Choose an option:
1. 🎯 Interactive Research Session
2. 📚 Run Example Research Queries  
3. 🔬 Single Research Query
4. 🚪 Exit

Enter your choice (1-4): 
```

---

## 🎯 Step 4: Usage Options

### Option 1: Interactive Research Session
```
Enter your choice (1-4): 1

🎯 Interactive Research Session Started
=====================================
💡 Type 'help' for commands, 'quit' to exit

Research Query: transformer architectures for multimodal learning
Domain: computer science
```

### Option 2: Example Research Queries
```
Enter your choice (1-4): 2

📚 Running Example Research Queries...
🔬 Query: Latest advances in quantum computing algorithms
🏷️  Domain: physics
```

### Option 3: Single Research Query
```
Enter your choice (1-4): 3

📝 Enter research query: neural network optimization techniques
🏷️  Research domain (default: computer science): machine learning
```

---

## 📁 Step 5: Output Files

The agent creates several output files:

```
📁 C:\Users\PLNAYAK\Documents\Agent_Fly\
├── 📄 memento_research_memory.json     # Memory bank
├── 📄 research_session_YYYYMMDD.md     # Research results
├── 📄 research_paper_YYYYMMDD.tex      # LaTeX source (if generated)
└── 📄 research_paper_YYYYMMDD.pdf      # PDF output (if compiled)
```

---

## 🔧 Troubleshooting

### Common Issues:

**1. Import Error: `ModuleNotFoundError: No module named 'agent'`**
```bash
# Solution: Verify Memento directory structure
dir Memento\client\agent.py
# Should exist. If not, re-clone Memento repository
```

**2. API Key Error**
```bash
# Solution: Check .env file
type .env
# Verify API keys are set correctly
```

**3. MCP Server Connection Issues**
```bash
# Solution: Check if server scripts exist
dir Memento\server\*.py
# Should show: search_tool.py, documents_tool.py, code_agent.py
```

**4. Permission Errors**
```bash
# Solution: Run as administrator or check file permissions
# Right-click Command Prompt -> "Run as administrator"
```

---

## 🎮 Interactive Commands

When in interactive mode, you can use:

- **`help`** - Show available commands
- **`memory`** - View stored research cases
- **`stats`** - Show agent statistics
- **`clear`** - Clear conversation history
- **`save <filename>`** - Save current session
- **`quit`** - Exit the session

---

## 📊 Example Session

```bash
C:\Users\PLNAYAK\Documents\Agent_Fly> python memento_library_research_agent.py

🧠 Memento Library Research Agent
================================

Choose an option:
1. 🎯 Interactive Research Session
2. 📚 Run Example Research Queries
3. 🔬 Single Research Query
4. 🚪 Exit

Enter your choice (1-4): 1

🎯 Interactive Research Session Started
=====================================
💡 Type 'help' for commands, 'quit' to exit

Research Query: latest advances in transformer architectures
Domain: computer science

🔍 Retrieving similar research experiences...
📚 Found 2 similar cases in memory bank
🤖 Connecting to Memento MCP servers...
✅ Connected to search_tool.py
✅ Connected to documents_tool.py
✅ Connected to code_agent.py

🧠 Meta-Planner: Breaking down research task...
⚡ Executor: Searching for recent transformer papers...
📄 Found 15 relevant papers on transformer architectures
🔬 Analyzing key innovations and trends...

✅ Research completed!
📄 Results saved to: research_session_20240115_143022.md
💾 Experience stored in memory bank
```

---

## 🔍 Verification Steps

1. **Check Python Path**: `python --version` (should be 3.8+)
2. **Verify Imports**: `python -c "import sys; print('\\n'.join(sys.path))"`
3. **Test Memento Import**: `python -c "import sys; sys.path.insert(0, 'Memento/client'); from agent import HierarchicalClient; print('✅ Import successful')"`
4. **Check API Keys**: `python -c "from dotenv import load_dotenv; import os; load_dotenv(); print('OpenAI:', bool(os.getenv('OPENAI_API_KEY'))); print('Google:', bool(os.getenv('GOOGLE_API_KEY')))"`

---

## 📚 Import Path Analysis

### Where the `agent` library is imported from:

The `agent` library is imported from the **Memento repository's client directory**:

#### Path Setup:
```python
# Line 25: Define the Memento path
MEMENTO_PATH = Path(__file__).parent / "Memento"

# Lines 26-27: Add Memento directories to Python path
sys.path.insert(0, str(MEMENTO_PATH / "client"))  # Adds: C:\Users\PLNAYAK\Documents\Agent_Fly\Memento\client
sys.path.insert(0, str(MEMENTO_PATH / "server"))  # Adds: C:\Users\PLNAYAK\Documents\Agent_Fly\Memento\server
```

#### Import Source:
```python
# Line 30: Import from the added path
from agent import HierarchicalClient, OpenAIBackend, trim_messages
```

This imports from: **`C:\Users\PLNAYAK\Documents\Agent_Fly\Memento\client\agent.py`**

#### What's imported:

1. **`HierarchicalClient`** - The main Memento client class that implements the hierarchical agent architecture
2. **`OpenAIBackend`** - Backend for OpenAI API integration  
3. **`trim_messages`** - Utility function for managing message context length

#### Dependencies:

For this import to work, you need:
- ✅ The **Memento repository** cloned in `C:\Users\PLNAYAK\Documents\Agent_Fly\Memento\`
- ✅ The **`agent.py`** file exists in `Memento\client\agent.py`
- ✅ All Memento's dependencies installed (OpenAI, MCP, etc.)

---

## 🎯 Key Features

### Memory-Based Learning
- Stores research experiences in `memento_research_memory.json`
- Retrieves similar past research cases (K=4 optimal)
- Learns from each session without model retraining

### Hierarchical Architecture
- **Meta-Planner**: Breaks down research queries into executable tasks
- **Executor**: Performs individual research tasks using MCP tools
- **Memory System**: Stores and retrieves research experiences

### MCP Tool Integration
- **Search Tool**: Web search and academic paper discovery
- **Documents Tool**: PDF processing and text extraction
- **Code Agent**: Code analysis and generation capabilities

### Research Capabilities
- Academic paper discovery and analysis
- Literature review generation
- Research gap identification
- LaTeX document generation
- PDF compilation support

---

The agent is now ready to run! It will use Memento's hierarchical architecture to conduct research, learn from each session, and generate comprehensive research papers using the memory-based learning approach. 