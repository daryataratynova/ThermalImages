import argparse
import csv
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def evaluate_and_save(ground_truth_json, detection_json, output_csv):
    cocoGt = COCO(ground_truth_json)
    cocoDt = cocoGt.loadRes(detection_json)

    cocoEval = COCOeval(cocoGt, cocoDt, 'keypoints')

    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    metrics_names = [
        'Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ]',
        'Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ]',
        'Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ]',
        'Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ]',
        'Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ]',
        'Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ]',
        'Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ]',
        'Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ]',
        'Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ]',
        'Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ]',
    ]

    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = ['Metric', 'Value']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for i, score in enumerate(cocoEval.stats):
            writer.writerow({'Metric': metrics_names[i], 'Value': score})

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate COCO Keypoint Detections')
    parser.add_argument('--ground-truth-json', required=True, help='Path to ground truth annotations in COCO format')
    parser.add_argument('--detection-json', required=True, help='Path to detected annotations in COCO format')
    parser.add_argument('--output-csv', required=True, help='Path to save evaluation metrics in CSV format')

    args = parser.parse_args()
    evaluate_and_save(args.ground_truth_json, args.detection_json, args.output_csv)
