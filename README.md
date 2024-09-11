# GroqShell

GroqShell is a command-line interface for interacting with Groq AI models. It allows users to easily send prompts to Groq models and receive responses directly in their terminal.

## Features

- Interact with Groq AI models from the command line
- Select from available Groq models
- Option to force JSON output
- Persistent model selection
- Interactive mode for continuous conversation
- Markdown formatting for responses
- Syntax highlighting for code blocks
- Command history support

## Installation

To install GroqShell, follow these steps:

1. Install the package:
   ```
   pip install groqshell-cli
   ```

## Usage

Before using GroqShell, make sure to set your Groq API key as an environment variable:

```
export GROQ_API_KEY='your-api-key-here'
```

Basic usage:

```
groqshell -p "Your prompt here"
```

Force JSON output:

```
groqshell -p "Your prompt here" -j
```

Select a different Groq model:

```
groqshell -m
```

## Options

- `-p`, `--prompt`: The prompt to send to the Groq AI model (required)
- `-j`, `--json`: Force JSON output
- `-c`, `--change`: Change the Groq model
- `-i`, `--info`: Get information about the currently selected model
- `-l`, `--list`: List available Groq models
- `-I`, `--interactive`: Enter interactive mode for continuous conversation

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.
