import os
import glob
import numpy as np
import json
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from tqdm import tqdm
from moviepy.video.io.VideoFileClip import VideoFileClip
import PIL.Image


def calculate_psnr(img1, img2):
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


def process_video_psnr(gt_path, pred_path):
    try:
        clip_gt = VideoFileClip(gt_path)
        clip_pred = VideoFileClip(pred_path)

        fps = min(clip_gt.fps, clip_pred.fps)
        duration = min(clip_gt.duration, clip_pred.duration)

        time_points = np.arange(0, duration, 1.0 / fps)

        video_psnrs = []

        for t in time_points:
            frame_gt = clip_gt.get_frame(t)
            frame_pred = clip_pred.get_frame(t)

            img_gt = PIL.Image.fromarray(frame_gt).resize((256, 256), PIL.Image.Resampling.BILINEAR)
            img_pred = PIL.Image.fromarray(frame_pred).resize((256, 256), PIL.Image.Resampling.BILINEAR)

            psnr = calculate_psnr(np.array(img_gt), np.array(img_pred))
            video_psnrs.append(psnr)

        clip_gt.close()
        clip_pred.close()

        return np.mean(video_psnrs) if video_psnrs else 0.0

    except Exception as e:
        print(f"Error processing {os.path.basename(gt_path)}: {e}")
        return None


def main():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--gt_video', type=str, required=True, help='path to reference videos')
    parser.add_argument('--pred_video', type=str, required=True, help='path to pred videos')
    parser.add_argument('--output_file', type=str, default=None, help='path to output file')
    args = parser.parse_args()

    if not os.path.exists(args.gt_video):
        print(f"Error: GT video not found at {args.gt_video}")
        return
    if not os.path.exists(args.pred_video):
        print(f"Error: Pred video not found at {args.pred_video}")
        return

    print(f"Comparing:\nRef: {args.gt_video}\nPred: {args.pred_video}")

    v_psnr = process_video_psnr(args.gt_video, args.pred_video)

    if v_psnr is not None:
        print("-" * 30)
        print(f"Video PSNR: {v_psnr:.4f} dB")
        print("-" * 30)

        if args.output_file:
            result = {
                "gt_video": args.gt_video,
                "pred_video": args.pred_video,
                "psnr": v_psnr
            }
            with open(args.output_file, 'w') as f:
                json.dump(result, f, indent=4)
            print(f"Result saved to {args.output_file}")
    else:
        print("Failed to calculate PSNR.")


if __name__ == '__main__':
    main()
