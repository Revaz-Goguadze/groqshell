import argparse
import json
import os
from groq import Groq
import readline
import atexit
import colorama
from colorama import Fore, Style
import re
import pygments
from pygments import highlight
from pygments.lexers import get_lexer_by_name
from pygments.formatters import TerminalFormatter


def check_api_key():
    if "GROQ_API_KEY" not in os.environ:
        print("Error: GROQ_API_KEY is not set in your environment.")
        print("Please set your Groq API key using:")
        print("export GROQ_API_KEY='your-api-key-here'")
        exit(1)


def select_groq_model():
    check_api_key()
    client = Groq()
    available_models = client.models.list()
    print("Available Groq models:")
    for i, model in enumerate(available_models.data, 1):
        print(f"{i}. {model.id}")

    while True:
        try:
            choice = int(input("Select a model number: "))
            if 1 <= choice <= len(available_models.data):
                selected_model = available_models.data[choice - 1].id
                save_selected_model(selected_model)
                return selected_model
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")


def save_selected_model(model):
    with open("selected_model.json", "w") as f:
        json.dump({"model": model}, f)


def load_selected_model():
    try:
        with open("selected_model.json", "r") as f:
            data = json.load(f)
            return data.get("model")
    except FileNotFoundError:
        return None


def change_model():
    print("Changing the model...")
    return select_groq_model()


def get_model_info(client, model_id):
    try:
        model = client.models.retrieve(model_id)
        print(f"Model '{model.id}' created by {model.owned_by}.")
    except Exception as e:
        print(f"Error retrieving model info: {str(e)}")
        return None


def list_available_models(client):
    try:
        models = client.models.list()
        return models.data
    except Exception as e:
        print(f"Error listing models: {str(e)}")
        return []


def generate_completion(client, model, messages, max_tokens=100):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating completion: {str(e)}")
        return None


def setup_history():
    histfile = os.path.join(os.path.expanduser("~"), ".groqshell_history")
    try:
        readline.read_history_file(histfile)
        readline.set_history_length(100000)
    except FileNotFoundError:
        pass
    atexit.register(readline.write_history_file, histfile)


def format_markdown(text):
    # Format headers
    for i in range(6, 0, -1):
        text = re.sub(
            r"^{} (.+)$".format("#" * i),
            lambda m: f"\n{Fore.CYAN}{Style.BRIGHT}{m.group(1)}{Style.RESET_ALL}",
            text,
            flags=re.MULTILINE,
        )

    # Format code blocks
    text = re.sub(
        r"```(\w+)?\n([\s\S]*?)```",
        lambda m: format_code_block(m.group(2), m.group(1)),
        text,
    )

    # Format inline code
    text = re.sub(
        r"`([^`\n]+)`", lambda m: f"{Fore.GREEN}{m.group(1)}{Style.RESET_ALL}", text
    )

    # Format bold text
    text = re.sub(
        r"\*\*(.*?)\*\*|__(.*?)__",
        lambda m: f"{Style.BRIGHT}{m.group(1) or m.group(2)}{Style.RESET_ALL}",
        text,
    )

    # Format italic text
    text = re.sub(
        r"\*(.*?)\*|_(.*?)_",
        lambda m: f"{Fore.YELLOW}{m.group(1) or m.group(2)}{Style.RESET_ALL}",
        text,
    )

    # Format strikethrough text
    text = re.sub(
        r"~~(.*?)~~", lambda m: f"{Style.DIM}{m.group(1)}{Style.RESET_ALL}", text
    )

    # Format blockquotes
    text = re.sub(
        r"^> (.+)$",
        lambda m: f"{Fore.LIGHTBLACK_EX}{m.group(1)}{Style.RESET_ALL}",
        text,
        flags=re.MULTILINE,
    )

    # Format horizontal rules
    text = re.sub(
        r"^([-*_])\1{2,}$",
        lambda m: f"\n{Fore.WHITE}{'-' * 40}{Style.RESET_ALL}\n",
        text,
        flags=re.MULTILINE,
    )

    # Format unordered lists
    text = re.sub(
        r"^([ \t]*)([-*+]) (.+)$",
        lambda m: f"{m.group(1)}{Fore.YELLOW}•{Style.RESET_ALL} {m.group(3)}",
        text,
        flags=re.MULTILINE,
    )

    # Format ordered lists
    text = re.sub(
        r"^([ \t]*)(\d+)\. (.+)$",
        lambda m: f"{m.group(1)}{Fore.YELLOW}{m.group(2)}.{Style.RESET_ALL} {m.group(3)}",
        text,
        flags=re.MULTILINE,
    )

    # Format links
    text = re.sub(
        r"\[([^\]]+)\]\(([^\)]+)\)",
        lambda m: f"{Fore.BLUE}{m.group(1)}{Style.RESET_ALL} ({Fore.CYAN}{m.group(2)}{Style.RESET_ALL})",
        text,
    )

    return text


def format_code_block(code, language=None):
    if language:
        try:
            lexer = get_lexer_by_name(language, stripall=True)
        except pygments.util.ClassNotFound:
            lexer = get_lexer_by_name("text", stripall=True)
    else:
        lexer = get_lexer_by_name("text", stripall=True)

    formatter = TerminalFormatter()
    highlighted_code = highlight(code, lexer, formatter)
    return f"\n{highlighted_code}\n"


def interactive_mode(client, selected_model):
    setup_history()
    colorama.init()
    print(
        f"{Fore.GREEN}Entering interactive mode. Type 'exit' or press Ctrl+D to quit.{Style.RESET_ALL}"
    )
    messages = []
    while True:
        try:
            prompt = input(f"{Fore.BLUE}groqshell> {Style.RESET_ALL}")
            if prompt.lower() == "exit":
                print("\nExiting...")
                break
            messages.append({"role": "user", "content": prompt})
            response = generate_completion(client, selected_model, messages)
            if response:
                formatted_response = format_markdown(response)
                print(f"{formatted_response}")
                messages.append({"role": "assistant", "content": response})
        except KeyboardInterrupt:
            print("\nKeyboard interrupt detected. Type 'exit' or press Ctrl+D to quit.")
        except EOFError:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")


def main():
    check_api_key()

    parser = argparse.ArgumentParser(description="Groq AI Shell Interface")
    parser.add_argument("-p", "--prompt", type=str, help="Prompt for Groq AI")
    parser.add_argument("-j", "--json", action="store_true", help="Force JSON output")
    parser.add_argument("-c", "--change", action="store_true", help="Change Groq model")
    parser.add_argument("-i", "--info", action="store_true", help="Get model info")
    parser.add_argument(
        "-l", "--list", action="store_true", help="List available models"
    )
    parser.add_argument(
        "-I", "--interactive", action="store_true", help="Enter interactive mode"
    )

    args = parser.parse_args()

    if not any([args.prompt, args.change, args.info, args.list, args.interactive]):
        parser.error(
            "At least one of -p/--prompt, -c/--change, -i/--info, -l/--list, or -I/--interactive is required"
        )

    client = Groq()

    if args.change:
        selected_model = change_model()
    else:
        selected_model = load_selected_model()

    if selected_model is None:
        selected_model = select_groq_model()

    if args.info:
        get_model_info(client, selected_model)

    if args.list:
        models = list_available_models(client)
        print("Available Groq models:")
        for model in models:
            print(f"- {model.id}")
        return

    if args.interactive:
        interactive_mode(client, selected_model)
        return

    if args.prompt:
        try:
            prompt = args.prompt
            response_format = None

            if args.json or "json" in prompt.lower():
                response_format = {"type": "json_object"}
                if "json" not in prompt.lower():
                    prompt += " Please provide the response in JSON format."

            stream = client.chat.completions.create(
                model=selected_model,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                response_format=response_format,
            )
            full_response = ""
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    full_response += content

            # Format the full response after it's complete
            formatted_response = format_markdown(full_response)
            print(f"\n{formatted_response}")
        except Exception as e:
            print(f"Error in Groq API call: {str(e)}")


if __name__ == "__main__":
    main()
