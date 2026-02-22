"""Minimal vLLM namespace for bench serve."""

import sys

sys.modules.setdefault("vllm", sys.modules[__name__])
