# Copyright (C) 2025 AIDC-AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Pixelle-Video LLM CLI

Usage:
    pvideo-llm "Explain atomic habits"
    echo "Explain..." | pvideo-llm
    pvideo-llm --preset qwen "Hello"
"""

import argparse
import asyncio
import json
import os
import sys

# loguru emits logs at import time because pixelle_video.__init__ eagerly
# loads ConfigManager. Suppress stderr during imports, restore afterwards.
_stderr_fd = sys.stderr.fileno()
_stderr_backup = os.dup(_stderr_fd)
_devnull = os.open(os.devnull, os.O_WRONLY)
os.dup2(_devnull, _stderr_fd)
os.close(_devnull)

from loguru import logger

from pixelle_video.services.llm_service import LLMService
from pixelle_video.llm_presets import get_preset_names, get_preset

# Restore stderr
os.dup2(_stderr_backup, _stderr_fd)
os.close(_stderr_backup)


def _reconfigure_stdout():
    """Reconfigure stdout for UTF-8 to handle emoji and non-ASCII chars on Windows."""
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


def main():
    """CLI entry point for LLM text generation."""
    _reconfigure_stdout()
    parser = argparse.ArgumentParser(
        prog="pvideo-llm",
        description="Generate text using LLM",
    )
    parser.add_argument("prompt", nargs="?", help="The prompt to generate from")
    parser.add_argument("-m", "--model", help="Model name")
    parser.add_argument("-k", "--api-key", help="API key")
    parser.add_argument("-u", "--base-url", help="API base URL")
    parser.add_argument("-t", "--temperature", type=float, default=0.7)
    parser.add_argument("-n", "--max-tokens", type=int, default=2000)
    parser.add_argument("-j", "--json", action="store_true", dest="json_output",
                        help="Output in JSON format (for agent consumption)")
    parser.add_argument("-c", "--config", help="Config file path")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    parser.add_argument("--list-presets", action="store_true",
                        help="List available LLM presets and exit")
    parser.add_argument("--preset", help="Use a preset provider (qwen/openai/claude/deepseek/ollama/moonshot)")

    args = parser.parse_args()

    if args.verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG", colorize=True)

    sys.exit(cmd_llm(args))


def cmd_llm(args) -> int:
    """Execute LLM call. Returns exit code."""
    if args.list_presets:
        for name in get_preset_names():
            preset = get_preset(name)
            print(f"{preset['name']}: {preset['model']} @ {preset['base_url']}")
        return 0

    # Resolve prompt: positional arg > stdin
    prompt = args.prompt
    if not prompt:
        if not sys.stdin.isatty():
            prompt = sys.stdin.read().strip()
        if not prompt:
            print("Error: No prompt provided. Usage: pvideo-llm <prompt>", file=sys.stderr)
            return 2

    # Resolve API parameters: CLI args > preset > config.yaml
    api_key = args.api_key
    base_url = args.base_url
    model = args.model

    if args.preset:
        preset = get_preset(args.preset)
        if not preset:
            print(f"Error: Unknown preset '{args.preset}'. Available: {', '.join(get_preset_names())}",
                  file=sys.stderr)
            return 2
        api_key = api_key or preset.get("default_api_key")
        base_url = base_url or preset.get("base_url")
        model = model or preset.get("model")

    # If config path specified, reset singleton and re-init
    if args.config:
        from pixelle_video.config.manager import ConfigManager
        ConfigManager._instance = None
        ConfigManager(args.config)

    # Create service and call
    service = LLMService({})

    try:
        result = asyncio.run(service(
            prompt=prompt,
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        ))
    except Exception as e:
        if args.json_output:
            output = json.dumps({
                "ok": False,
                "error": str(e),
                "model": model or "unknown",
            }, ensure_ascii=False)
            print(output)
        else:
            print(f"Error: {e}", file=sys.stderr)
        return 1

    # Output result
    if args.json_output:
        output = json.dumps({
            "ok": True,
            "text": result,
            "model": model or service.active,
        }, ensure_ascii=False)
        print(output)
    else:
        print(result)

    return 0


if __name__ == "__main__":
    sys.exit(main())
