import argparse

from pathlib import Path
from frontveg.core.pipeline import FrontVegPipeline

def main():
    parser = argparse.ArgumentParser(description="FrontVeg Segmentation Pipeline")
    parser.add_argument("--input", type=str, required=True, help="Path to raw images")
    parser.add_argument("--output", type=str, default="output_frontveg", help="Output directory")
    parser.add_argument("--encoder", type=str, default="vitl", choices=["vits", "vitb", "vitl", "vitg"])
    parser.add_argument("--plot", action="store_true", help="Enable histogram plotting")
    parser.add_argument("--depth_repo_path", type=str, default="external/Depth-Anything-V2", help="Path to Depth Anything repository")

    parser.add_argument("--sigma", type=float, default=1.1, help="Sigma for Gaussian smoothing in valley detection")
    parser.add_argument("--peak_dist", type=float, default=2, help="Minimum distance between peaks in histogram")
    parser.add_argument("--peak_height", type=float, default=0.5, help="Minimum height of peaks in histogram")
    parser.add_argument("--smoothed", action="store_true", help="Enable smoothing in valley detection")
    args = parser.parse_args()

    # sigma=1.1, peak_dist=2, peak_height=0.5
  
    config = {
        'input': args.input,
        'output': args.output,
        'encoder': args.encoder,
        'depth_repo_path': args.depth_repo_path,
        'sigma': args.sigma,
        'peak_dist': args.peak_dist,
        'peak_height': args.peak_height,
        'smoothed': args.smoothed
    }

    # Initialize and run
    pipeline = FrontVegPipeline(config)
    
    # import os
        # for subdir in os.listdir(args.input):
        #     full_subdir = os.path.join(args.input, subdir)
        #     if os.path.isdir(full_subdir):
        #         pipeline.process_folder(full_subdir, args.output, plot_graphics=args.plot)

    input_path = Path(args.input)
    subdirs = [p for p in input_path.iterdir() if p.is_dir()]

    if subdirs:
        for d in subdirs:
            pipeline.process_folder(str(d), args.output, plot_graphics=args.plot)
    else:
        pipeline.process_folder(str(input_path), args.output, plot_graphics=args.plot)

if __name__ == "__main__":
    main()



# # PYTHONPATH=src python -m frontveg.main_frontveg \
# --input data/raw/arbo_test/test \
# --output outputs/experiments \
# --encoder vitl \
# --plot \
# --peak_dist 1