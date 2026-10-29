# Human Detection in Video

Detects and counts people in a video using a pre-trained deep-learning object detector, drawing a bounding box around each person and writing an annotated output video. It's the detection building block behind the [Public Security System](https://github.com/ankitamullagiri/Public-Security-System-using-Deep-Learning) project.

## How it works

1. Load a **frozen inference graph** from the TensorFlow 1 Detection Model Zoo, trained on COCO (80 classes, where class 1 = person).
2. Read the video frame by frame with OpenCV and resize each frame to 1280×720.
3. Run the detector to get bounding boxes, confidence scores and class IDs.
4. Keep only `person` detections above a confidence threshold (default 0.7), draw boxes, and overlay a live people count.
5. Write the annotated frames to `output.mp4` and report processing FPS.

No training is involved. This project is about applying and wrapping a pre-trained model for inference.

## Setup

```bash
pip install -r requirements.txt
```

Download a COCO-trained detector from the [TF1 Detection Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md), for example `faster_rcnn_inception_v2_coco` (more accurate) or `ssd_mobilenet_v2_coco` (faster). Extract it and use the `frozen_inference_graph.pb` inside.

## Run

```bash
python human_detection.py --model frozen_inference_graph.pb --video input.mp4
python human_detection.py --model frozen_inference_graph.pb --video input.mp4 --threshold 0.6 --show   # live preview, press q to quit
```

## Trade-offs

| Choice | Effect |
|---|---|
| Higher threshold | Fewer false positives, more missed people in crowds or poor lighting |
| Faster R-CNN vs SSD MobileNet | Better accuracy vs real-time speed on CPU |
| 1280×720 resize | Consistent input size, but small distant people get harder to detect |

## Next steps

- Port to a TF2 SavedModel or YOLOv8 for faster inference.
- Add tracking (e.g. SORT/DeepSORT) to count unique people rather than per-frame detections.

## Tech stack

Python · TensorFlow · OpenCV · NumPy
