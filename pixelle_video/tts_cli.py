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
Pixelle-Video TTS CLI

Usage:
    pvideo-tts "你好世界" --local
    pvideo-tts "Hello" --comfyui --workflow tts_edge.json
    echo "text" | pvideo-tts --local
"""

import argparse
import asyncio
import json
import sys

from pixelle_video.service import PixelleVideoCore
from pixelle_video.services.tts_service import TTSService
from pixelle_video.tts_voices import EDGE_TTS_VOICES


def main():
    """CLI entry point for TTS text-to-speech generation."""
    parser = argparse.ArgumentParser(
        prog="pvideo-tts",
        description="Generate speech from text using TTS",
    )
    parser.add_argument("prompt", nargs="?", help="Text to convert to speech")
    parser.add_argument("-m", "--mode", required=True, choices=["local", "comfyui"],
                        help="Inference mode: local (Edge TTS) or comfyui (workflow)")
    parser.add_argument("--voice", help="Voice ID (default: zh-CN-YunjianNeural)")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Speech speed multiplier (default: 1.0)")
    parser.add_argument("--workflow", help="Workflow filename (comfyui mode)")
    parser.add_argument("--comfyui-url", help="ComfyUI URL (overrides config)")
    parser.add_argument("--runninghub-api-key", help="RunningHub API key")
    parser.add_argument("-j", "--json", action="store_true", dest="json_output",
                        help="Output in JSON format (for agent consumption)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    parser.add_argument("--list-voices", action="store_true",
                        help="List available voices and exit")

    args = parser.parse_args()
    sys.exit(cmd_tts(args))


def cmd_tts(args) -> int:
    """Execute TTS call. Returns exit code."""
    if args.list_voices:
        for v in EDGE_TTS_VOICES:
            print(f"{v['id']}  ({v['locale']}, {v['gender']})")
        return 0

    # Resolve text: positional arg > stdin
    text = args.prompt
    if not text:
        if not sys.stdin.isatty():
            text = sys.stdin.read().strip()
        if not text:
            print("Error: No text provided. Usage: pvideo-tts <text>", file=sys.stderr)
            return 2

    core = PixelleVideoCore()
    service = TTSService(core.config, core=core)

    try:
        result = asyncio.run(service(
            text=text,
            inference_mode=args.mode,
            voice=args.voice,
            speed=args.speed,
            workflow=args.workflow,
            comfyui_url=args.comfyui_url,
            runninghub_api_key=args.runninghub_api_key,
        ))
    except Exception as e:
        if args.json_output:
            output = json.dumps({
                "ok": False,
                "error": str(e),
            }, ensure_ascii=False)
            print(output)
        else:
            print(f"Error: {e}", file=sys.stderr)
        return 1

    if args.json_output:
        output = json.dumps({
            "ok": True,
            "path": result,
        }, ensure_ascii=False)
        print(output)
    else:
        print(result)

    return 0


if __name__ == "__main__":
    sys.exit(main())
