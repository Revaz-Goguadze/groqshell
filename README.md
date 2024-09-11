# GroqShell

Welcome to **GroqShell**, a powerful command-line interface designed for seamless interaction with Groq AI models. With GroqShell, you can effortlessly send prompts to Groq models and receive instant responses right in your terminal.

## 🌟 Features

- **Command-Line Interaction**: Engage with Groq AI models directly from your terminal.
- **Model Selection**: Choose from a variety of available Groq models.
- **JSON Output**: Easily force JSON output for structured responses.
- **Persistent Model Selection**: Your selected model is remembered for future sessions.
- **Interactive Mode**: Enjoy a continuous conversation with the AI.
- **Markdown Formatting**: Responses are beautifully formatted in Markdown.
- **Syntax Highlighting**: Code blocks are highlighted for better readability.
- **Command History Support**: Navigate through your command history effortlessly.

## 🚀 Installation

To get started with GroqShell, follow these simple steps:

1. Install the package using pip:

   ```bash
   pip install groqshell-cli
   ```

## Usage

Before using **GroqShell**, make sure to set your **Groq API key** as an environment variable:

```
export GROQ_API_KEY='your-api-key-here'
```

**Basic usage:**

```
groqshell -p "Your prompt here"
```

**Force JSON output:**

```
groqshell -p "Your prompt here" -j
```

**Select a different Groq model:**

```
groqshell -m
```

**List available Groq models:**

```
groqshell -l
```

**Enter interactive mode:**

```
groqshell -I
```

## 🔍 Options

- `-p`, `--prompt`: The prompt to send to the Groq AI model (required)
- `-j`, `--json`: Force JSON output
- `-c`, `--change`: Change the Groq model
- `-i`, `--info`: Get information about the currently selected model
- `-l`, `--list`: List available Groq models
- `-I`, `--interactive`: Enter interactive mode for continuous conversation

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📜 License

This project is licensed under the MIT License.
