# Turtle.AI-coder

A web server for loading and running custom AI models, inspired by llama.cpp's server interface.

## Features

- 🐢 **Modern Web Interface**: Beautiful, responsive UI with dark theme
- 🤖 **Model Management**: Load/unload custom AI models dynamically
- ⚙️ **Configurable Settings**: Adjust temperature, max tokens, and context size
- 💬 **Chat Interface**: Full chat functionality with message history
- 🔌 **OpenAI-Compatible API**: Compatible with OpenAI's completion and chat completion endpoints
- 📡 **Streaming Support**: Real-time token streaming for completions
- ❤️ **Health Monitoring**: Built-in health check endpoint

## Quick Start

### Run the Server

```bash
# Basic usage (default: http://127.0.0.1:8080)
python3 turtle_ai_coder.py

# With custom port
python3 turtle_ai_coder.py --port 8080

# Pre-load a model on startup
python3 turtle_ai_coder.py --model /path/to/your/model.gguf

# Custom configuration
python3 turtle_ai_coder.py --host 0.0.0.0 --port 8080 --context-size 4096
```

### Command Line Options

```
--host HOST           Host to bind to (default: 127.0.0.1)
--port PORT           Port to bind to (default: 8080)
--model MODEL         Path to model file to load on startup
--context-size SIZE   Context size (default: 4096)
--threads NUM         Number of threads (default: 4)
```

## API Endpoints

### Web Interface
- `GET /` - Main web interface

### Health & Info
- `GET /health` - Server health check
- `GET /models` - List loaded models
- `GET /settings` - Get current settings

### Model Control
- `POST /load_model` - Load a custom model
- `POST /unload_model` - Unload current model

### Completion APIs
- `POST /completion` - Text completion (llama.cpp style)
- `POST /chat/completions` - Chat completion (OpenAI style)

## API Examples

### Load a Model

```bash
curl -X POST http://127.0.0.1:8080/load_model \
  -H "Content-Type: application/json" \
  -d '{"model": "/path/to/your/model.gguf"}'
```

### Text Completion

```bash
curl -X POST http://127.0.0.1:8080/completion \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Write a function to calculate factorial",
    "temperature": 0.7,
    "max_tokens": 512
  }'
```

### Chat Completion

```bash
curl -X POST http://127.0.0.1:8080/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are a helpful coding assistant."},
      {"role": "user", "content": "How do I reverse a string in Python?"}
    ],
    "temperature": 0.7
  }'
```

### Streaming Completion

```bash
curl -X POST http://127.0.0.1:8080/completion \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain quantum computing",
    "stream": true
  }'
```

### Update Settings

```bash
curl -X POST http://127.0.0.1:8080/settings \
  -H "Content-Type: application/json" \
  -d '{
    "temperature": 0.8,
    "max_tokens": 1024,
    "context_size": 4096
  }'
```

## Integrating Your Custom Model

The current implementation includes a placeholder `generate_response()` function. To integrate your own trained model:

1. Locate the `generate_response()` function in `turtle_ai_coder.py`
2. Replace the placeholder logic with your model's inference code
3. Ensure your model loading logic is implemented in `serve_load_model()`

Example integration point:

```python
def generate_response(prompt, temperature=0.7, max_tokens=512):
    """
    Replace this with your actual model inference code.
    """
    # Your model inference here
    # Example:
    # from your_model import Model
    # model = Model.load(model_state.model_path)
    # response = model.generate(prompt, temperature, max_tokens)
    # return response
    
    # Current placeholder
    return "Your model's response here"
```

## Project Structure

```
/workspace/
├── turtle_ai_coder.py    # Main server script
├── README.md             # This file
└── ...
```

## License

See LICENSE file in the repository.

## Acknowledgments

- Inspired by [llama.cpp](https://github.com/ggerganov/llama.cpp) server interface
- UI design inspired by modern AI chat interfaces
