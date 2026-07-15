import argparse
import logging
from pipelines.video_pipeline import VideoInjuryPipeline
from pipelines.realtime_pipeline import RealtimePipeline

logging.basicConfig(level=logging.INFO)


def main():
    parser = argparse.ArgumentParser(description="Sports Injury Detection")
    parser.add_argument("--mode",   choices=["video", "realtime"], default="video")
    parser.add_argument("--input",  type=str, default="data/raw/sample.mp4",
                        help="Input video path (video mode)")
    parser.add_argument("--output", type=str, default="outputs/videos/result.mp4",
                        help="Output video path")
    parser.add_argument("--source", type=int, default=0,
                        help="Webcam index or RTSP URL (realtime mode)")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    if args.mode == "video":
        pipeline = VideoInjuryPipeline(config_path=args.config)
        events = pipeline.run(input_path=args.input, output_path=args.output)
        print(f"\n✅ Analysis complete. {len(events)} injury events logged.")

    elif args.mode == "realtime":
        pipeline = RealtimePipeline(config_path=args.config, source=args.source)
        pipeline.run()


if __name__ == "__main__":
    main()