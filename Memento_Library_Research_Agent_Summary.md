# Memento Library Research Agent

## Overview

The `memento_library_research_agent.py` is a sophisticated AI-powered research paper generation system that leverages Memento's hierarchical agent architecture to automate academic research workflows. This implementation demonstrates advanced AI agent design patterns including memory-based learning, tool orchestration, and multi-format output generation.

## Key Features

### 🧠 **Hierarchical Agent Architecture**
- **Meta-Planner**: Strategic task decomposition using configurable LLM models (default: O3-mini)
- **Executor**: Tactical task execution with tool orchestration capabilities
- **Dual-model design**: Separate models for planning and execution phases

### 💾 **Memory-Based Learning System**
- **Case Memory**: Stores and retrieves similar research experiences (optimal K=4 retrieval)
- **Similarity Matching**: Uses query and domain overlap for case selection
- **Learning Loop**: Captures success metrics, tool usage, and insights for continuous improvement
- **Persistent Storage**: JSON-based case bank with timestamp and complexity tracking

### 🛠️ **MCP Tool Integration**
- **SearxNG Search**: Privacy-respecting literature search and web research
- **Document Processing**: PDF extraction and analysis (planned)
- **Code Generation**: Analysis scripts and LaTeX formatting
- **Modular Design**: Easy addition of new research tools

### 📄 **Multi-Format Output System**
- **Always generates 4 files**: `.md`, `.tex`, `.html`, and `.pdf` (when available)
- **Smart Content Detection**: Automatically identifies LaTeX vs. Markdown content
- **Format Conversion**: Bidirectional LaTeX ↔ Markdown conversion
- **PDF Generation**: Tectonic LaTeX engine with fallback to browser printing

## Technical Implementation

### Core Classes

**`MementoResearchAgent`**
- Main orchestrator class
- Configurable LLM models
- MCP server connection management
- Research paper generation workflow

**`MementoMemorySystem`**
- Case-based learning implementation
- Similarity scoring algorithm
- JSON persistence layer
- Research case storage and retrieval

**`ResearchCase`** (DataClass)
- Query, domain, and approach tracking
- Tool usage and success metrics
- Insights and timestamp recording
- Task complexity measurement

### Processing Pipeline

1. **Memory Retrieval**: Query similar past research cases
2. **Context Enhancement**: Augment query with memory insights
3. **Hierarchical Processing**: Meta-planner creates JSON task plan
4. **Task Execution**: Executor uses MCP tools for research tasks
5. **Content Processing**: Smart format detection and conversion
6. **Multi-format Generation**: Create all output formats
7. **Learning Storage**: Record case for future reference

### Error Handling & Robustness

- **Graceful Degradation**: System continues when components fail
- **Alternative Workflows**: Multiple PDF generation strategies
- **Tool Failure Recovery**: Missing tools don't crash the system
- **Memory Resilience**: Continues operation even with case loading failures

## Performance Characteristics

- **Initialization**: 2-5 seconds (MCP server connection)
- **Research Generation**: 30-120 seconds (complexity-dependent)
- **Memory Usage**: ~200-500MB during operation
- **Storage**: ~10-50KB per research paper set

## Usage Examples

### Basic Research Generation
```python
agent = MementoResearchAgent()
await agent.initialize()

result = await agent.generate_research_paper(
    "Mixture of Agents for Enhanced Reasoning",
    domain="artificial intelligence"
)
```

### Custom Model Configuration
```python
agent = MementoResearchAgent(
    meta_model="llama-3.3-70b-versatile",
    exec_model="o3-mini"
)
```

### Interactive Research Session
```python
await agent.interactive_research_session()
```

## Dependencies

- **Core**: `asyncio`, `json`, `pathlib`, `dataclasses`
- **AI Models**: OpenAI API, GROQ API support
- **MCP**: Model Context Protocol for tool integration
- **LaTeX**: Tectonic engine for PDF generation
- **Environment**: `python-dotenv` for configuration

## File Structure

```
memento_library_research_agent.py (992 lines)
├── LaTeX PDF Rendering (38-99)
├── Memory System (105-181)
├── Research Prompts (186-221)
├── Main Agent Class (227-500)
├── Content Conversion (600-850)
└── CLI Interface (915-992)
```

## Innovation Highlights

1. **Memory-Enhanced Planning**: Uses past research experiences to inform new research strategies
2. **Smart Format Handling**: Automatically detects and converts between LaTeX and Markdown
3. **Tool Orchestration**: Seamless integration of research tools via MCP protocol
4. **Learning System**: Captures and reuses successful research patterns
5. **Production Ready**: Robust error handling and graceful degradation

## Future Enhancements

- Enhanced tool ecosystem (document processing, image analysis)
- Advanced similarity matching with semantic embeddings
- Multi-agent collaboration for complex research projects
- Domain-specific research templates
- Quality assessment and validation systems

---

**Technical Complexity**: Advanced  
**Lines of Code**: 992  
**Architecture Pattern**: Hierarchical Agent with Memory  
**Primary Use Case**: Automated Academic Research Paper Generation  
**Key Innovation**: Memory-based learning without model retraining
