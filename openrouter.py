
import logging
import argparse
import os
import sys
from dataclasses import dataclass
import requests


try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

#  models from OpenRouter
AVAILABLE_MODELS = [
    "openai/gpt-4o",
    "meta-llama/llama-3.1-70b-instruct",
    "meta-llama/llama-3.1-8b-instruct",
    "mistralai/mistral-large",
    "mistralai/mistral-nemo",
    "google/gemini-pro-1.5",
    "anthropic/claude-3.5-sonnet-20240620",
    "cohere/command-r-plus",
    
    # can add more
]

logger = logging.getLogger(__name__)

class ApiKeyNotFoundException(Exception):
    pass


class InvalidSelectedModelException(Exception):
    pass


def display(
    message: str,
    color: str = "white",
    bold: bool = False,
    bg_color: str = "black",
):
    colors = {
        "red": "91m",
        "green": "92m",
        "yellow": "93m",
        "blue": "94m",
        "white": "97m",
    }
    bg_colors = {
        "black": "40",
        "red": "41",
        "green": "42",
        "yellow": "43",
        "blue": "44",
        "white": "47",
    }
    if bold:
        print(f"\033[1;{bg_colors[bg_color]};{colors[color]} {message}\033[0m")
    else:
        print(f"\033[{bg_colors[bg_color]};{colors[color]} {message}\033[0m")


@dataclass(frozen=True)
class ApiConfig:
    api_url: str = "https://openrouter.ai/api/v1/chat/completions"
    api_key: str | None = None
    usage: bool = False
    model: str | None = None


class ModelValidator:
    @staticmethod
    def validate(model: str) -> bool:
        return model in AVAILABLE_MODELS

    @staticmethod
    def get_AVAILABLE_MODELS() -> list[str]:
        return AVAILABLE_MODELS


class ApiKeyValidator:
    @staticmethod
    def get_api_key_from_system() -> str | None:
        return os.environ.get("OPENROUTER_API_KEY")


class OpenRouter:
    def __init__(self, args) -> None:
        self.setup = ApiConfig
        if not ModelValidator.validate(args.model):
            raise InvalidSelectedModelException(
                f"Invalid model: {args.model}\n"
                f"Available models: {ModelValidator.get_AVAILABLE_MODELS()}"
            )
        self.setup.model = args.model
        self.setup.usage = args.usage
        self.use_glow = args.glow
        if not args.api_key:
            api_key = ApiKeyValidator.get_api_key_from_system()
            if api_key is None:
                display("API key not found in environment variables! ", "red")
                logger.debug("API key not found in environment variables!")
                raise ApiKeyNotFoundException
            else:
                logger.debug(f"API key found in environment variables")
                self.setup.api_key = api_key
        else:
            self.setup.api_key = args.api_key

    def get_response(self, message) -> None:
        headers = {
            "Authorization": f"Bearer {self.setup.api_key}",
            "HTTP-Referer": "https://github.com/your-repo",  # Required by OpenRouter
            "X-Title": "OpenRouter CLI",  # Optional
            "Content-Type": "application/json"
        }
        logger.debug(f"Headers: {headers}")
        
        query_data = {
            "model": self.setup.model,
            "messages": [
                {"role": "system", "content": "Be precise and concise."},
                {"role": "user", "content": message},
            ],
            "max_tokens": 1000,  # Limit tokens for free-tier compatibility
        }
        logger.debug(f"Query data: {query_data}")

        try:
            response = requests.post(
                self.setup.api_url, 
                headers=headers, 
                json=query_data,
                timeout=30
            )
            response.raise_for_status()

            result = response.json()
            if self.setup.usage:
                self._show_usage(result.get("usage", {}), self.use_glow)
            self._show_content(result["choices"][0]["message"]["content"])
            
        except requests.exceptions.HTTPError as err:
            if response.status_code == 401:
                display("Invalid API key! ", "red")
            else:
                display(f"HTTP Error: {err}", "red")
            logger.error(f"HTTP Error: {err}\nResponse: {response.text}")
            sys.exit(1)
        except requests.exceptions.RequestException as err:
            display(f"Request Error: {err}", "red")
            logger.error(f"Request Error: {err}")
            sys.exit(1)
        except KeyError as err:
            display(f"Unexpected response format: {err}", "red")
            logger.error(f"KeyError: {err}\nResponse: {response.text}")
            sys.exit(1)

    @staticmethod
    def _show_usage(result: dict, use_glow: bool) -> None:
        if use_glow:
            print("# Tokens")
        else:
            display("Tokens \n", "yellow", True, "blue")
        for token in result:
            print(f"- {token}: {result[token]}")
        print("\n")

    def _show_content(self, result: str) -> None:
        if self.use_glow:
            print("# Response")
        else:
            display("Response \n", "yellow", True, "blue")
        print(result)


def main() -> None:
    parser = argparse.ArgumentParser(description='OpenRouter CLI Client')
    parser.add_argument("query", type=str, help="The query to process")
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug mode")
    parser.add_argument("-u", "--usage", action="store_true", help="Show token usage")
    parser.add_argument("-g", "--glow", action="store_true", help="Simple output formatting")
    parser.add_argument(
        "-a",
        "--api-key",
        type=str,
        help="OpenRouter API key (default: uses OPENROUTER_API_KEY environment variable)",
        required=False,
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        help=f"Model to use (default: openai/gpt-3.5-turbo). Available models: {AVAILABLE_MODELS}",
        required=False,
        default="openai/gpt-3.5-turbo",
    )
    args = parser.parse_args()
    log_level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger.debug(f"args: {args}")
    try:
        openrouter = OpenRouter(args)
        openrouter.get_response(args.query)
    except Exception as e:
        logger.debug(f"An error occurred: {str(e)}")
        display(f"Error: {str(e)}", "red")
        sys.exit(1)


if __name__ == "__main__":
    main()