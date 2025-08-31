# Memento: Quick Setup Guide

## 🚀 Quick Start (Windows)

### Prerequisites Checklist
- [ ] Python 3.10+ installed
- [ ] Git installed
- [ ] Docker Desktop installed and running
- [ ] OpenAI API key ready

### Step 1: Clone and Setup Environment

```cmd
# Clone the repository
git clone https://github.com/Agent-on-the-Fly/Memento.git
cd Memento

# Create and activate conda environment
conda create -n Memento python=3.11 -y
conda activate Memento
```

### Step 2: Install Dependencies

```cmd
# Install core requirements
pip install -r requirements.txt

# Install web crawling tools
pip install -U crawl4ai
crawl4ai-setup
crawl4ai-doctor

# Install Playwright browsers
playwright install
```

### Step 3: Configure Environment

Navigate to `client/` directory and create `.env` file:

```cmd
cd client
```

Create `.env` file with the following content:

```env
# Required: OpenAI Configuration
OPENAI_API_KEY=sk-your-openai-api-key-here
OPENAI_BASE_URL=https://api.openai.com/v1

# Optional: Additional Tool APIs
CHUNKR_API_KEY=your_chunkr_api_key_here
JINA_API_KEY=your_jina_api_key_here
ASSEMBLYAI_API_KEY=your_assemblyai_api_key_here
```

### Step 4: Setup SearxNG (Web Search)

Open a **new terminal** and run:

```cmd
cd Memento\searxng-docker
docker compose up -d
```

Verify SearxNG is running by visiting: http://localhost:8080

### Step 5: Test Installation

Return to your main terminal in the `Memento/client` directory:

```cmd
python agent.py
```

You should see the Memento agent start up and prompt for input.

---

## 🔧 Configuration Options

### Model Configuration

Edit the model settings in `client/agent.py`:

```python
# Line 55: Change executor model
EXE_MODEL = "o3"  # Options: "o3", "gpt-4", "gpt-4-turbo"

# For planner model, modify the HierarchicalAgent initialization
planner_model = "gpt-4.1"  # Options: "gpt-4.1", "gpt-4", "claude-3-opus"
```

### SearxNG Configuration

To use a different SearxNG instance, edit `server/search_tool.py`:

```python
# Line 31: Change default host
DEFAULT_HOST = "http://your-searxng-instance.com"
```

### Tool Server Configuration

Each tool server runs independently. To customize:

1. **Code Execution Security** (`server/code_agent.py`):
   ```python
   # Add allowed packages to the whitelist
   DEFAULT_IMPORT_WHITELIST.extend([
       'your_custom_package',
       'another_package'
   ])
   ```

2. **Search Results** (`server/search_tool.py`):
   ```python
   # Modify default parameters
   DEFAULT_RESULTS = 15  # More search results
   DEFAULT_CATEGORY = "science"  # Different category
   ```

---

## 🧪 Testing Your Setup

### Test 1: Basic Query
```
Query: "What is the current weather in New York?"
Expected: Uses search tool to find current weather information
```

### Test 2: Code Execution
```
Query: "Calculate the fibonacci sequence for the first 10 numbers"
Expected: Uses code execution to generate and display the sequence
```

### Test 3: Document Processing
```
Query: "Create a simple Python script and save it to a file"
Expected: Uses code agent to write and save a Python file
```

---

## 🐛 Common Issues and Solutions

### Issue 1: SearxNG Not Starting
```cmd
# Check Docker is running
docker ps

# Restart SearxNG
cd searxng-docker
docker compose down
docker compose up -d

# Check logs
docker compose logs searxng
```

### Issue 2: OpenAI API Errors
```
Error: "Invalid API key"
Solution: 
1. Check your .env file has correct OPENAI_API_KEY
2. Ensure no extra spaces or quotes around the key
3. Verify the key is active on OpenAI dashboard
```

### Issue 3: Import Errors
```
Error: "Module not found"
Solution:
1. Ensure conda environment is activated: conda activate Memento
2. Reinstall requirements: pip install -r requirements.txt
3. For missing packages: pip install package_name
```

### Issue 4: Playwright Installation
```cmd
# If playwright browsers fail to install
playwright install --force

# For Windows-specific issues
pip install --upgrade playwright
playwright install chromium
```

---

## 📊 Usage Examples

### Example 1: Research Query

**Input:**
```
What are the main differences between React and Vue.js in 2024?
```

**Expected Workflow:**
1. Meta-planner breaks down into search tasks
2. Executor searches for React vs Vue comparisons
3. Synthesizes information into comprehensive comparison

### Example 2: Data Analysis

**Input:**
```
Create a Python script that generates a bar chart of the top 5 programming languages by popularity
```

**Expected Workflow:**
1. Planner identifies need for code creation and execution
2. Executor writes Python script using matplotlib
3. Executes script and shows results

### Example 3: Document Processing

**Input:**
```
Help me understand what tools are available in this system
```

**Expected Workflow:**
1. Planner identifies need to explore available tools
2. Executor lists and describes MCP tools
3. Provides comprehensive overview of capabilities

---

## 🔄 Development Workflow

### Adding New Tools

1. **Create Tool Server:**
   ```python
   # server/my_new_tool.py
   from mcp.server.fastmcp import FastMCP
   
   mcp = FastMCP("my_tool")
   
   @mcp.tool()
   def my_function(param: str) -> str:
       """Description of what this tool does"""
       return f"Processed: {param}"
   ```

2. **Register in Agent:**
   ```python
   # client/agent.py
   # Add to server list in HierarchicalAgent.__init__()
   ("my_tool", ["python", "server/my_new_tool.py"])
   ```

### Testing Changes

```cmd
# Run agent with verbose logging
python agent.py --verbose

# Test specific tool
python server/my_new_tool.py --test
```

---

## 📈 Performance Optimization

### Memory Usage
- Monitor token usage with `tiktoken`
- Implement context window management
- Use appropriate model selection for tasks

### Response Time
- Cache frequently used results
- Implement parallel tool execution
- Optimize search parameters

### Cost Management
- Use smaller models for simple tasks
- Implement request batching
- Monitor API usage

---

## 🎯 Next Steps

1. **Explore Advanced Features:**
   - Custom tool development
   - Memory system integration (when available)
   - Multi-modal processing

2. **Integrate with Your Workflow:**
   - API integration
   - Custom UI development
   - Automation scripts

3. **Contribute:**
   - Report issues on GitHub
   - Submit tool improvements
   - Share usage examples

---

## 📚 Additional Resources

- **GitHub Repository:** https://github.com/Agent-on-the-Fly/Memento
- **Research Paper:** arxiv.org/abs/2508.16153
- **SearxNG Documentation:** https://docs.searxng.org/
- **MCP Protocol:** https://modelcontextprotocol.io/

---

*Happy building with Memento! 🎉* 