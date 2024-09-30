import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import torch
from PIL import Image

from imagenet_classes import IMAGENET2012_CLASSES


def parse_arguments():
    parser = argparse.ArgumentParser(description="Video inference with TensorRT")
    parser.add_argument("--output_video", type=str, help="Path to output video file")
    parser.add_argument("--input_video", type=str, help="Path to input video file")
    parser.add_argument("--webcam", action="store_true", help="Use webcam as input")
    parser.add_argument(
        "--live", action="store_true", help="View video live during inference"
    )
    return parser.parse_args()


def get_ort_session(model_path):
    providers = [
        (
            "TensorrtExecutionProvider",
            {
                "device_id": 0,
                "trt_max_workspace_size": 8589934592,
                "trt_fp16_enable": True,
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": "./trt_cache",
                "trt_force_sequential_engine_build": False,
                "trt_max_partition_iterations": 10000,
                "trt_min_subgraph_size": 1,
                "trt_builder_optimization_level": 5,
                "trt_timing_cache_enable": True,
            },
        ),
    ]
    return ort.InferenceSession(model_path, providers=providers)


def preprocess_frame(frame):
    # Use cv2 for resizing instead of PIL for better performance
    resized = cv2.resize(frame, (448, 448), interpolation=cv2.INTER_LINEAR)
    img_numpy = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
    img_numpy = img_numpy.transpose(2, 0, 1)
    img_numpy = np.expand_dims(img_numpy, axis=0)
    return img_numpy


def get_top_predictions(output, top_k=5):
    output = torch.from_numpy(output)
    probabilities, class_indices = torch.topk(output.softmax(dim=1) * 100, k=top_k)
    im_classes = list(IMAGENET2012_CLASSES.values())
    class_names = [im_classes[i] for i in class_indices[0]]
    return list(zip(class_names, probabilities[0].tolist()))


def draw_predictions(frame, predictions, fps):
    # Draw FPS in the top right corner with dark blue background
    fps_text = f"FPS: {fps:.2f}"
    (text_width, text_height), _ = cv2.getTextSize(
        fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
    )
    text_offset_x = frame.shape[1] - text_width - 10
    text_offset_y = 30
    box_coords = (
        (text_offset_x - 5, text_offset_y + 5),
        (text_offset_x + text_width + 5, text_offset_y - text_height - 5),
    )
    cv2.rectangle(
        frame, box_coords[0], box_coords[1], (139, 0, 0), cv2.FILLED
    )  # Dark blue background
    cv2.putText(
        frame,
        fps_text,
        (text_offset_x, text_offset_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),  # White text
        2,
    )

    # Draw predictions
    for i, (name, prob) in enumerate(predictions):
        text = f"{name}: {prob:.2f}%"
        cv2.putText(
            frame,
            text,
            (10, 30 + i * 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

    # Draw model name at the bottom of the frame with red background
    model_name = "Model: eva02_large_patch14_448"
    (text_width, text_height), _ = cv2.getTextSize(
        model_name, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
    )
    text_x = (frame.shape[1] - text_width) // 2
    text_y = frame.shape[0] - 20
    box_coords = (
        (text_x - 5, text_y + 5),
        (text_x + text_width + 5, text_y - text_height - 5),
    )
    cv2.rectangle(
        frame, box_coords[0], box_coords[1], (0, 0, 255), cv2.FILLED
    )  # Red background
    cv2.putText(
        frame,
        model_name,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),  # White text
        2,
    )

    return frame


def process_video(input_path, output_path, session, live_view=False, use_webcam=False):
    if use_webcam:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(input_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    frame_count = 0
    total_time = 0
    current_fps = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.time()

        preprocessed = preprocess_frame(frame)
        output = session.run([output_name], {input_name: preprocessed})
        predictions = get_top_predictions(output[0])

        end_time = time.time()
        frame_time = end_time - start_time
        current_fps = 1 / frame_time

        frame_with_predictions = draw_predictions(frame, predictions, current_fps)

        if out:
            out.write(frame_with_predictions)

        if live_view:
            cv2.imshow("Inference", frame_with_predictions)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        total_time += frame_time
        frame_count += 1

        print(
            f"Processed frame {frame_count}, Time: {frame_time:.3f}s, FPS: {current_fps:.2f}"
        )

    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()

    avg_time = total_time / frame_count
    print(f"Average processing time per frame: {avg_time:.3f}s")
    print(f"Average FPS: {1/avg_time:.2f}")


def main():
    args = parse_arguments()
    session = get_ort_session("merged_model_compose.onnx")

    if args.webcam:
        process_video(None, args.output_video, session, args.live, use_webcam=True)
    elif args.input_video:
        process_video(args.input_video, args.output_video, session, args.live)
    else:
        print("Error: Please specify either --input_video or --webcam")
        return


if __name__ == "__main__":
    main()
