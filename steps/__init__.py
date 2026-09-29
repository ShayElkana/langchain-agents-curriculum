from dotenv import load_dotenv
import os

# Load environment variables from .env at program load
load_dotenv()


def main() -> None:
    api_key = os.environ.get("OPENAI_API_KEY")
    print("Hello from agents-test1!")
