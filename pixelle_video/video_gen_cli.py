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
Pixelle-Video Video Generation CLI

Usage:
    pvideo-video-gen "sunset beach" --workflow video_wan.json --duration 5
    pvideo-video-gen "a drone flyover" -w video_wan.json --steps 30
    echo "underwater scene" | pvideo-video-gen -w video_wan.json
"""

import argparse
import asyncio
import json
import sys

from pixelle_video.service import PixelleVideoCore
from pixelle_video.services.media import MediaService


def main():
    """CLI entry point for AI video generation."""
    parser = argparse.ArgumentParser(
        prog="pvideo-video-gen",
        description="Generate videos from text using ComfyUI workflows",
    )
    parser.add_argument("prompt", nargs="?", help="Video generation prompt")
    parser.add_argument("-w", "--workflow", help="Workflow filename (e.g. video_wan.json)")
    parser.add_argument("--width", type=int, help="Video width")
    parser.add_argument("--height", type=int, help="Video height")
    parser.add_argument("--duration", type=float, help="Video duration in seconds")
    parser.add_argument("--steps", type=int, help="Sampling steps")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument("--cfg", type=float, help="CFG scale (prompt adherence)")
    parser.add_argument("--sampler", help="Sampler name")
    parser.add_argument("--negative-prompt", help="Negative prompt")
    parser.add_argument("-u", "--comfyui-url", help="ComfyUI URL (overrides config)")
    parser.add_argument("--runninghub-api-key", help="RunningHub API key")
    parser.add_argument("-o", "--output", dest="output_path", help="Output file path")
    parser.add_argument("-j", "--json", action="store_true", dest="json_output",
                        help="Output in JSON format (for agent consumption)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()
    sys.exit(cmd_video_gen(args))


def cmd_video_gen(args) -> int:
    """Execute video generation. Returns exit code."""
    prompt = args.prompt
    if not prompt:
        if not sys.stdin.isatty():
            prompt = sys.stdin.read().strip()
        if not prompt:
            print("Error: No prompt provided. Usage: pvideo-video-gen <prompt>", file=sys.stderr)
            return 2

    core = PixelleVideoCore()
    service = MediaService(core.config, core=core)

    try:
        result = asyncio.run(service(
            prompt=prompt,
            media_type="video",
            workflow=args.workflow,
            comfyui_url=args.comfyui_url,
            runninghub_api_key=args.runninghub_api_key,
            width=args.width,
            height=args.height,
            duration=args.duration,
            steps=args.steps,
            seed=args.seed,
            cfg=args.cfg,
            sampler=args.sampler,
            negative_prompt=args.negative_prompt,
        ))
    except Exception as e:
        if args.json_output:
            output = json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False)
            print(output)
        else:
            print(f"Error: {e}", file=sys.stderr)
        return 1

    path = result.url
    if args.json_output:
        output = json.dumps({
            "ok": True,
            "path": path,
            "media_type": result.media_type,
        }, ensure_ascii=False)
        print(output)
    else:
        print(path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
