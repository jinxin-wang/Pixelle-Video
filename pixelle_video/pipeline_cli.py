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
Pixelle-Video Pipeline CLI - End-to-end video generation workflow

Usage:
    pvideo-pipeline "介绍AI的发展历史"
    pvideo-pipeline "机器学习科普" -n 3 --tts-voice zh-CN-XiaoxiaoNeural
    pvideo-pipeline --task-id mytask --resume
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from pixelle_video.service import PixelleVideoCore


def main():
    """CLI entry point for end-to-end video generation pipeline."""
    parser = argparse.ArgumentParser(
        prog="pvideo-pipeline",
        description="Generate videos from topic using full pipeline (narrations -> TTS -> media -> compose)",
    )
    parser.add_argument("topic", nargs="?", help="Video topic or script")
    parser.add_argument("-p", "--pipeline", default="standard",
                        choices=["standard", "custom", "asset_based"],
                        help="Pipeline type (default: standard)")
    parser.add_argument("-n", "--n-scenes", type=int, default=5,
                        help="Number of scenes (default: 5)")
    parser.add_argument("-o", "--output", dest="output_path",
                        help="Output video file path")
    parser.add_argument("--title", help="Video title (auto-generated if not provided)")

    # TTS options
    tts = parser.add_argument_group("TTS options")
    tts.add_argument("--tts-mode", default="local", choices=["local", "comfyui"],
                     help="TTS inference mode (default: local)")
    tts.add_argument("--tts-voice", default="zh-CN-YunjianNeural",
                     help="Voice ID (default: zh-CN-YunjianNeural)")
    tts.add_argument("--tts-speed", type=float, default=1.2,
                     help="Speech speed (default: 1.2)")
    tts.add_argument("--tts-workflow", help="TTS workflow filename (comfyui mode)")

    # Media options
    media = parser.add_argument_group("Media generation options")
    media.add_argument("--media-type", choices=["image", "video"],
                        help="Media type: generates static images or video clips per frame "
                             "(auto-selects frame template and workflow)")
    media.add_argument("--media-workflow", help="Media workflow filename")
    media.add_argument("--media-width", type=int, help="Media width")
    media.add_argument("--media-height", type=int, help="Media height")
    media.add_argument("--frame-template",
                        help="HTML frame template (auto-set by --media-type if not specified)")

    # BGM options
    bgm = parser.add_argument_group("BGM options")
    bgm.add_argument("--bgm", dest="bgm_path", help="Background music file path")
    bgm.add_argument("--bgm-volume", type=float, default=0.2,
                     help="BGM volume 0.0-1.0 (default: 0.2)")

    parser.add_argument("-j", "--json", action="store_true", dest="json_output",
                        help="Output in JSON format (for agent consumption)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    parser.add_argument("--task-id", help="Task ID for grouping outputs under output/<task_id>/")
    parser.add_argument("--resume", action="store_true",
                        help="Skip frames with existing output files (requires --task-id)")

    args = parser.parse_args()
    sys.exit(cmd_pipeline(args))


def _get_task_dir(task_id: str) -> str:
    """Ensure output/{task_id}/ exists and return its path."""
    task_dir = Path("output") / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    return str(task_dir)


def cmd_pipeline(args) -> int:
    """Execute video generation pipeline. Returns exit code."""
    if args.resume and not args.task_id:
        print("Error: --resume requires --task-id", file=sys.stderr)
        return 2

    topic = args.topic
    if not topic:
        if not sys.stdin.isatty():
            topic = sys.stdin.read().strip()
        if not topic:
            if args.resume:
                # In resume mode, try to read topic from existing narrations.json
                task_dir = _get_task_dir(args.task_id)
                narrations_file = Path(task_dir) / "narrations.json"
                if narrations_file.exists():
                    data = json.loads(narrations_file.read_text(encoding="utf-8"))
                    topic = data.get("text", data.get("prompt", ""))
                if not topic:
                    print("Error: --resume requires a topic, either via arg/stdin or existing narrations.json",
                          file=sys.stderr)
                    return 2
            else:
                print("Error: No topic provided. Usage: pvideo-pipeline <topic>", file=sys.stderr)
                return 2

    # Resolve output path
    output_path = args.output_path
    if args.task_id and not output_path:
        task_dir = _get_task_dir(args.task_id)
        output_path = str(Path(task_dir) / "final.mp4")

    # Resolve media type: auto-select frame template
    frame_template = args.frame_template
    if args.media_type and not frame_template:
        frame_template = f"1080x1920/{args.media_type}_default.html"

    async def run():
        async with PixelleVideoCore() as core:
            return await core.generate_video(
                text=topic,
                pipeline=args.pipeline,
                n_scenes=args.n_scenes,
                output_path=output_path,
                title=args.title,
                tts_inference_mode=args.tts_mode,
                tts_voice=args.tts_voice,
                tts_speed=args.tts_speed,
                tts_workflow=args.tts_workflow,
                media_workflow=args.media_workflow,
                media_width=args.media_width,
                media_height=args.media_height,
                frame_template=frame_template,
                bgm_path=args.bgm_path,
                bgm_volume=args.bgm_volume,
                resume=args.resume,
                task_id=args.task_id,
            )

    try:
        result = asyncio.run(run())
    except Exception as e:
        if args.json_output:
            output = json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False)
            print(output)
        else:
            print(f"Error: {e}", file=sys.stderr)
        return 1

    if args.json_output:
        output = {
            "ok": True,
            "video_path": result.video_path,
            "duration": result.duration,
            "file_size": result.file_size,
            "title": result.storyboard.title,
            "n_frames": len(result.storyboard.frames),
        }
        if args.task_id:
            output["task_id"] = args.task_id
        print(json.dumps(output, ensure_ascii=False))
    else:
        print(result.video_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
