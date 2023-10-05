import json
import os
from pathlib import Path
import argparse

def find_route(data, id):
    return [route for route in data if route.get('id') == id]

def labelstudio_labels_to_yolo(labelstudio_labels_path: str, label_names_path: str, output_dir_path: str, id: int) -> None:
    with open(label_names_path, 'r') as f:
        label_names = f.read().split('\n')
    print('Label names:', label_names)
    with open(labelstudio_labels_path, 'r') as f:
        labelstudio_labels_json = f.read()

    # print(len(json.loads(labelstudio_labels_json)))
    labels = json.loads(labelstudio_labels_json)
    # every box stores the frame count of the whole video so we get it from the first box
    # 
    
    # iterate through boxes

    labels = find_route(labels, id)[0]

    frames_count = labels['annotations'][0]['result'][0]['value']['framesCount']
    yolo_labels = [[] for _ in range(frames_count)]

    for box in labels['annotations'][0]['result']:
        label_numbers = [label_names.index(label) for label in box['value']['labels']]
        # iterate through keypoints (we omit the last keypoint because no interpolation after that)
        for i, keypoint in enumerate(box['value']['sequence'][:-1]):
            start_point = keypoint
            end_point = box['value']['sequence'][i + 1]
            start_frame = start_point['frame']
            end_frame = end_point['frame']

            n_frames_between = end_frame - start_frame
            delta_x = (end_point['x'] - start_point['x']) / n_frames_between
            delta_y = (end_point['y'] - start_point['y']) / n_frames_between
            delta_width = (end_point['width'] - start_point['width']) / n_frames_between
            delta_height = (end_point['height'] - start_point['height']) / n_frames_between

            # In YOLO, x and y are in the center of the box. In Label Studio, x and y are in the corner of the box.
            x = start_point['x'] + start_point['width'] / 2
            y = start_point['y'] + start_point['height'] / 2
            width = start_point['width']
            height = start_point['height']
            # iterate through frames between two keypoints
            for frame in range(start_frame, end_frame):
                # Support for multilabel
                yolo_labels = _append_to_yolo_labels(yolo_labels, frame, label_numbers, x, y, width, height)
                x += delta_x + delta_width / 2
                y += delta_y + delta_height / 2
                width += delta_width
                height += delta_height
            # Make sure that the loop works as intended
            epsilon = 1e-5
            assert (x - end_point['x'] - end_point['width'] / 2) <= epsilon, f'x does not match: {x} vs {end_point["x"] + end_point["width"] / 2}'
            assert (y - end_point['y'] - end_point['height'] / 2) <= epsilon, f'y does not match: {y} vs {end_point["y"] + end_point["height"] / 2}'
            assert (width - end_point[
                'width']) <= epsilon, f'width does not match: {width} vs {end_point["width"]}'
            assert (height - end_point[
                'height']) <= epsilon, f'height does not match: {height} vs {end_point["height"]}'

        # Handle last keypoint
        yolo_labels = _append_to_yolo_labels(yolo_labels, frame, label_numbers, x, y, width, height)
        for label_number in label_numbers:
            # frame-1 because Label Studio index starts from 1
            yolo_labels[frame-1].append(
                [label_number, x / 100, y / 100, width / 100, height / 100])

    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)
        print(f'Directory did not exist. Created {output_dir_path}')
    for frame, frame_labels in enumerate(yolo_labels):
        if frame % 1000 == 0:
            print(f'Writing labels for frame {frame}')
        padded_frame_number = str(frame).zfill(len(str(len(yolo_labels))))
        file_path = Path(output_dir_path) / f'frame_{padded_frame_number}.txt'
        text = ''
        for label in frame_labels:
            text += ' '.join(map(str, label)) + '\n'
        with open(file_path, 'w') as f:
            f.write(text)
    print(f'Done. Wrote labels for {frame + 1} frames.')


def _append_to_yolo_labels(yolo_labels: list, frame: int, label_numbers: list, x, y, width, height):
    for label_number in label_numbers:
        # current_frame-1 because Label Studio index starts from 1
        yolo_labels[frame-1].append(
            [label_number, x / 100, y / 100, width / 100, height / 100])
    return yolo_labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', help='Path of the .txt file containing Label Studio labels.', required=True)
    parser.add_argument('--names', '-n', help='Path of the .json file containing (case sensitive) label names.',
                        required=True)
    parser.add_argument('--output', '-o', help='Path of the output directory of .txt files containing YOLO labels.',
                        required=True)
    parser.add_argument('--id', '-f', help='Specify Project\'s ID',
                        required=True)

    args = parser.parse_args()

    labelstudio_labels_to_yolo(args.input, args.names, args.output, int(args.id))
