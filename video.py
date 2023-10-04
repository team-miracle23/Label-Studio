import os
import cv2
from tqdm import tqdm
from pathlib import Path
import argparse

def video_2_images(video_path: str, images_dir_path: str, target_frame_rate: float = 25):
    def save_frame(frame_number, img):
        padded_frame_number = str(frame_number).zfill(len(str(target_length)))
        save_path = Path(images_dir_path) / f'frame_{padded_frame_number}.jpg'
        cv2.imwrite(str(save_path), img)

    if not os.path.exists(images_dir_path):
        os.makedirs(images_dir_path)
        print(f'Directory did not exist. Created {images_dir_path}')
    cap = cv2.VideoCapture(video_path)

    orig_frame_rate = cap.get(cv2.CAP_PROP_FPS)
    orig_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target_length = round(orig_length / orig_frame_rate * target_frame_rate)

    print(f'Original frames: {orig_length}')
    print(f'Target frames: {target_length}')

    pbar = tqdm(total=target_length)

    frame_id = 0
    success, frame = cap.read()

    save_frame(frame_id, frame)

    last_position_s = 0
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        position_s = float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000
        delta_s = position_s - last_position_s
        if delta_s >= (1 / target_frame_rate):
            frame_id += 1
            save_frame(frame_id, frame)
            pbar.update(1)
            last_position_s += 1 / target_frame_rate

    cap.release()
    print(f'Done. Wrote video from {video_path} to {images_dir_path}. Last frame was {frame_id}.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', help='Path of the video file.',
                        required=True)
    parser.add_argument('--output', '-o', help='Path of the output directory of .jpg files.',
                        required=True)
    parser.add_argument('--frame-rate', '-fr', help='Target framerate.', required=True)
    args = parser.parse_args()

    video_2_images(args.input, args.output, float(args.frame_rate))
