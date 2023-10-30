import argparse
from ultralytics import YOLO
import json
import os

def main(args):
    model = YOLO(args.model_path)

    with open(args.ground_truth, 'r') as f:
        ground_truth_data = json.load(f)

    file_name_to_id = {img['file_name']: img['id'] for img in ground_truth_data['images']}

    results = model(args.image_folder)
    json_results = []

    for result in results:
        image_id = file_name_to_id.get(os.path.basename(result.path))

        if image_id is None:
            print(f"Image ID not found for path: {result.path}")
            continue

        num_keypoints = len(result.keypoints.xy) if result.keypoints else 0
        num_boxes = len(result.boxes.xyxy) if result.boxes else 0

        if num_keypoints > 0 and num_boxes > 0 and num_keypoints == num_boxes:
            for i in range(num_keypoints):
                xy_coords = result.keypoints.xy[i].cpu().numpy().flatten().tolist()
                conf_scores = result.keypoints.conf[i].cpu().numpy().flatten().tolist()

                keypoints = []
                for x, y, c in zip(xy_coords[::2], xy_coords[1::2], conf_scores):
                    keypoints.extend([x, y, c])

                bbox = result.boxes.xyxy[i].cpu().numpy().flatten().tolist()
                score = result.boxes.conf[i].cpu().numpy().tolist()

                detection_json = {
                    "id": len(json_results),
                    "iscrowd": 0,
                    "image_id": image_id,
                    "category_id": 1,
                    "keypoints": keypoints,
                    "num_keypoints": len(keypoints) // 3,
                    "bbox": bbox,
                    "score": score,
                }

                json_results.append(detection_json)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(json_results, f)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument('--ground_truth', type=str, help='Path to the ground truth JSON file')
    parser.add_argument('--image_folder', type=str, help='Path to the folder containing images')
    parser.add_argument('--model_path', type=str, default='yolov8x-pose-p6.pt', help='Path to the YOLO model')
    parser.add_argument('--output', type=str, default='detections.json', help='Output JSON file')
    args = parser.parse_args()
    main(args)