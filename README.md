

---

# OpenRouter-CLI 

A lightweight command-line interface (CLI) to interact with cutting-edge AI models (GPT-4o, Claude 3.5, Llama 3, etc.) via [OpenRouter](https://openrouter.ai/). Perfect for developers, researchers, and AI enthusiasts.



## Features

- Query **Multiple AI models** with a single command
- **No API wrappers** – Direct HTTP requests for minimal latency
- **Token usage stats** (`--usage` flag)
- Colorful terminal output (or minimalist `--glow` mode)
- Zero-config setup (just add your API key)
- Built-in model validation

## Quick Start

### 1. Install
```bash
git clone git@github.com:Rishabh23112/OpenRouter.ai-CLI.git
cd Open-Router_CLI
pip install -e . 
```

### 2. Configure API Key
```bash
export OPENROUTER_API_KEY="your-key-here"  
```

### 3. Run!
```bash
python3 openrouter.py "Your Prompt here" --model anthropic/claude-3.5-sonnet #model name
```

## Full Usage

### Basic Query
```bash
python3 openrouter.py "Your prompt here"
```

### Advanced Options
| Flag               | Description                          | Example                                |
|--------------------|--------------------------------------|----------------------------------------|
| `-m/--model`       | Choose AI model                      | `-m meta-llama/llama-3.1-70b-instruct` |
| `-u/--usage`       | Show token usage                     | `-u`                                   |
| `-g/--glow`        | Minimalist output                    | `-g`                                   |
| `-a/--api-key`     | Pass API key manually                | `-a "sk-..."`                          |
| `-v/--verbose`     | Debug mode                           | `-v`                                   |

### Available Models
```text
openai/gpt-4o
anthropic/claude-3.5-sonnet
meta-llama/llama-3.1-70b-instruct
mistralai/mistral-large
google/gemini-pro-1.5
...and more!
```


## 🌟 Pro Tips

1. **Create an alias** (add to `~/.bashrc`):
   ```bash
   alias ai="openrouter-cli -m anthropic/claude-3.5-sonnet"
   ```
   Then just run: `ai "Your question"`

2. **Pipe inputs**:
   ```bash
   cat script.py | openrouter-cli "Debug this Python code"
   ```

3. **Save outputs**:
   ```bash
   openrouter-cli "Generate Django models" > models.py
   ```

## 🛠️ Development

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
pytest

# Format code
black .
```

## ❓ FAQ

**Q: Where do I get an API key?**  
A: Sign up at [OpenRouter.ai](https://openrouter.ai/keys)

**Q: How to update the model list?**  
A: Edit `AVAILABLE_MODELS` in [openrouter.py](openrouter.py)

## 📜 License
MIT © Rishabh Kumar  

---

