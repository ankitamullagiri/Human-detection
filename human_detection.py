"""Detect people in a video with a pre-trained TensorFlow object detector.

Loads a frozen inference graph from the TensorFlow 1 Detection Model Zoo
(COCO-trained, where class 1 = person), runs it frame by frame, draws a box
around each person above a confidence threshold and writes an annotated video.

Usage:
    python human_detection.py --model frozen_inference_graph.pb --video input.mp4
    python human_detection.py --model frozen_inference_graph.pb --video input.mp4 --show
"""

import argparse
import time

import cv2
import numpy as np
import tensorflow.compat.v1 as tf

PERSON_CLASS_ID = 1  # COCO label map


class DetectorAPI:
    """Thin wrapper around a frozen TF1 object-detection graph."""

    def __init__(self, path_to_ckpt: str):
        tf.disable_v2_behavior()
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            graph_def = tf.GraphDef()
            with tf.io.gfile.GFile(path_to_ckpt, "rb") as fid:
                graph_def.ParseFromString(fid.read())
                tf.import_graph_def(graph_def, name="")
        self.sess = tf.Session(graph=self.detection_graph)

        get = self.detection_graph.get_tensor_by_name
        self.image_tensor = get("image_tensor:0")
        self.detection_boxes = get("detection_boxes:0")
        self.detection_scores = get("detection_scores:0")
        self.detection_classes = get("detection_classes:0")
        self.num_detections = get("num_detections:0")

    def process_frame(self, image: np.ndarray):
        """Return pixel boxes (top, left, bottom, right), scores, classes and count."""
        boxes, scores, classes, num = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: np.expand_dims(image, axis=0)},
        )
        h, w = image.shape[:2]
        pixel_boxes = [
            (int(b[0] * h), int(b[1] * w), int(b[2] * h), int(b[3] * w)) for b in boxes[0]
        ]
        return pixel_boxes, scores[0].tolist(), [int(c) for c in classes[0]], int(num[0])

    def close(self):
        self.sess.close()


def run(model_path: str, video_path: str, output_path: str, threshold: float, show: bool) -> None:
    detector = DetectorAPI(model_path)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (1280, 720))

    frames, start = 0, time.time()
    while True:
        ok, img = cap.read()
        if not ok:
            break
        img = cv2.resize(img, (1280, 720), interpolation=cv2.INTER_AREA)
        boxes, scores, classes, _ = detector.process_frame(img)

        people = 0
        for (top, left, bottom, right), score, cls in zip(boxes, scores, classes):
            if cls == PERSON_CLASS_ID and score > threshold:
                people += 1
                cv2.rectangle(img, (left, top), (right, bottom), (255, 0, 0), 2)
        cv2.putText(img, f"People: {people}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

        writer.write(img)
        frames += 1
        if show:
            cv2.imshow("Human detection", img)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    elapsed = time.time() - start
    print(f"Processed {frames} frames in {elapsed:.1f}s ({frames / max(elapsed, 1e-9):.1f} FPS) -> {output_path}")
    cap.release()
    writer.release()
    detector.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect people in a video.")
    parser.add_argument("--model", required=True, help="path to frozen_inference_graph.pb")
    parser.add_argument("--video", required=True, help="input video file")
    parser.add_argument("--output", default="output.mp4", help="annotated output video")
    parser.add_argument("--threshold", type=float, default=0.7, help="minimum confidence (default 0.7)")
    parser.add_argument("--show", action="store_true", help="display frames while processing")
    args = parser.parse_args()
    run(args.model, args.video, args.output, args.threshold, args.show)
