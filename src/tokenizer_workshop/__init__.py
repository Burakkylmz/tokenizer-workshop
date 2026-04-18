from tokenizer_workshop.config import load_config


def main() -> None:
    config = load_config()

    print("Tokenizer Workshop")
    print(f"Project: {config.project.name}")
    print(f"Version: {config.project.version}")
    print(f"Debug: {config.project.debug}")
    print(f"Default tokenizer: {config.tokenizers.default_tokenizer}")
    print(f"First version tokenizers: {config.tokenizers.first_version}")
    print(f"LLM provider: {config.llm.provider}")
    print(f"LLM model: {config.llm.model}")
    print(f"GROQ_API_KEY loaded: {config.llm.api_key is not None}")