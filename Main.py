import argparse
import json
import os
from groq import Groq
import readline
import atexit
import colorama
from colorama import Fore, Style


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
        return model
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
        readline.set_history_length(1000)
    except FileNotFoundError:
        pass
    atexit.register(readline.write_history_file, histfile)


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
                print(f"{Fore.YELLOW}{response}{Style.RESET_ALL}")
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
    parser.add_argument("-m", "--model", action="store_true", help="Select Groq model")
    parser.add_argument("-c", "--change", action="store_true", help="Change Groq model")
    parser.add_argument("-i", "--info", action="store_true", help="Get model info")
    parser.add_argument(
        "-l", "--list", action="store_true", help="List available models"
    )
    parser.add_argument(
        "-I", "--interactive", action="store_true", help="Enter interactive mode"
    )

    args = parser.parse_args()

    if not any(
        [args.prompt, args.model, args.change, args.info, args.list, args.interactive]
    ):
        parser.error(
            "At least one of -p/--prompt, -m/--model, -c/--change, -i/--info, -l/--list, or -I/--interactive is required"
        )

    client = Groq()

    if args.model:
        selected_model = select_groq_model()
    elif args.change:
        selected_model = change_model()
    else:
        selected_model = load_selected_model()

    if selected_model is None:
        selected_model = select_groq_model()

    if args.info:
        model_info = get_model_info(client, selected_model)
        if model_info:
            print(f"Model Info for {selected_model}:")
            print(json.dumps(model_info, indent=2))
        return

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
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    print(chunk.choices[0].delta.content, end="", flush=True)
            print()
        except Exception as e:
            print(f"Error in Groq API call: {str(e)}")


if __name__ == "__main__":
    main()
