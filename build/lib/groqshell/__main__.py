import argparse
import json
import os
from groq import Groq


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


def main():
    check_api_key()

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Groq AI Shell Interface")
    parser.add_argument(
        "-p", "--prompt", type=str, required=True, help="Prompt for Groq AI"
    )
    parser.add_argument("-j", "--json", action="store_true", help="Force JSON output")
    parser.add_argument("-m", "--model", action="store_true", help="Select Groq model")

    # Parse arguments
    args = parser.parse_args()

    # Select or load Groq model
    if args.model:
        selected_model = select_groq_model()
    else:
        selected_model = load_selected_model()

    if selected_model is None:
        selected_model = select_groq_model()

    # Initialize Groq client
    client = Groq()

    try:
        # Prepare the prompt and response format
        prompt = args.prompt
        response_format = None

        if args.json or "json" in prompt.lower():
            response_format = {"type": "json_object"}
            if "json" not in prompt.lower():
                prompt += " Please provide the response in JSON format."

        # Generate response in chunks
        stream = client.chat.completions.create(
            model=selected_model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            response_format=response_format,
        )
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                print(chunk.choices[0].delta.content, end="", flush=True)
        print()  # Print a newline at the end
    except Exception as e:
        print(f"Error in Groq API call: {str(e)}")


if __name__ == "__main__":
    main()
