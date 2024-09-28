import time
import argparse
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import torch
from PIL import Image

from imagenet_classes import IMAGENET2012_CLASSES

def parse_arguments():
    parser = argparse.ArgumentParser(description="Video inference with TensorRT")
    parser.add_argument("input_video", type=str, help="Path to input video file")
    parser.add_argument("output_video", type=str, help="Path to output video file")
    parser.add_argument("--live", action="store_true", help="View video live during inference")
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
    # Draw FPS in the top right corner
    cv2.putText(frame, f"FPS: {fps:.2f}", (frame.shape[1] - 150, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Draw predictions
    for i, (name, prob) in enumerate(predictions):
        text = f"{name}: {prob:.2f}%"
        cv2.putText(frame, text, (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Draw model name at the bottom of the frame
    model_name = "Model: eva02_large_patch14_448"
    text_size = cv2.getTextSize(model_name, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    text_x = (frame.shape[1] - text_size[0]) // 2
    text_y = frame.shape[0] - 20
    cv2.putText(frame, model_name, (text_x, text_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return frame

def process_video(input_path, output_path, session, live_view=False):
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
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
        
        out.write(frame_with_predictions)
        
        if live_view:
            cv2.imshow('Inference', frame_with_predictions)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        total_time += frame_time
        frame_count += 1

        print(f"Processed frame {frame_count}, Time: {frame_time:.3f}s, FPS: {current_fps:.2f}")

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    avg_time = total_time / frame_count
    print(f"Average processing time per frame: {avg_time:.3f}s")
    print(f"Average FPS: {1/avg_time:.2f}")

def main():
    args = parse_arguments()
    session = get_ort_session("merged_model_compose.onnx")
    process_video(args.input_video, args.output_video, session, args.live)

if __name__ == "__main__":
    main()