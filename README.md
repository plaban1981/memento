Let me check the previous conversation and documentation to find the exact commands used for running SearxNG.

[1 tool called]

Based on the search results, here are the exact commands used for running SearxNG:

## 🐳 **SearxNG Docker Commands**

### **Primary Commands Used:**

#### **1. Navigate to SearxNG Directory:**
```bash
cd Memento/searxng-docker
```

#### **2. Start SearxNG (Main Command):**
```bash
docker compose up -d
```

#### **3. Stop SearxNG:**
```bash
docker compose down
```

#### **4. Check SearxNG Status:**
```bash
docker ps
```

#### **5. View SearxNG Logs:**
```bash
docker compose logs searxng
```

#### **6. Restart SearxNG (if needed):**
```bash
docker compose down
docker compose up -d
```

### **PowerShell Testing Commands Used:**

#### **Test SearxNG Connectivity:**
```powershell
# Test if SearxNG is running on localhost:8080
Invoke-WebRequest -Uri "http://127.0.0.1:8080" -Method GET | Select-Object -First 5
```

#### **Check Docker Container Status:**
```powershell
docker ps | findstr searxng
```

### **Configuration Details:**

#### **Default SearxNG Configuration:**
- **Port**: 8080 (mapped to localhost:8080)
- **Container Name**: `searxng`
- **Image**: `docker.io/searxng/searxng:latest`
- **Network**: Internal Docker network + port mapping

#### **Docker Compose Services:**
```yaml
services:
  searxng:
    container_name: searxng
    image: docker.io/searxng/searxng:latest
    ports:
      - "8080:8080"  # Maps container port 8080 to host port 8080
    networks:
      - searxng
```

### **Troubleshooting Commands:**

#### **If SearxNG Fails to Start:**
```bash
# Check Docker daemon
docker --version

# View detailed logs
docker compose logs -f searxng

# Rebuild containers
docker compose down
docker compose up -d --build

# Check port conflicts
netstat -ano | findstr :8080  # Windows
lsof -i :8080                 # Linux/Mac
```

#### **Verify SearxNG is Working:**
```bash
# Simple curl test
curl http://localhost:8080

# Or with PowerShell
Invoke-WebRequest http://localhost:8080
```

### **Integration with Memento Agent:**

The SearxNG server runs on `localhost:8080` and is accessed by the Memento research agent through:

```python
# In Memento/server/search_tool.py
DEFAULT_HOST = "http://127.0.0.1:8080"
```

### **Complete Setup Sequence:**

```bash
# 1. Navigate to directory
cd C:\Users\nayak\Documents\Agent_Fly\Memento\searxng-docker

# 2. Start SearxNG
docker compose up -d

# 3. Verify it's running
docker ps

# 4. Test connectivity (PowerShell)
Invoke-WebRequest -Uri "http://127.0.0.1:8080"

# 5. Run the research agent
cd ..\..\
python memento_library_research_agent.py
```

The key command that makes everything work is `docker compose up -d` from within the `Memento/searxng-docker` directory! 🚀
