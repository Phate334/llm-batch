import os

def get_adapter_absolute_path(lora_path: str) -> str:
    """
    Resolves the given lora_path to an absolute local path.
    """
    if os.path.isabs(lora_path):
        return lora_path

    if lora_path.startswith("~"):
        return os.path.expanduser(lora_path)

    if os.path.exists(lora_path):
        return os.path.abspath(lora_path)

    # For huggingface hub, we can just return the path and let transformers handle it
    return lora_path
