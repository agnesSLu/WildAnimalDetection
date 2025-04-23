import gradio as gr
from ultralytics import YOLO
import tempfile
import cv2
import numpy as np
from PIL import Image

MODEL_PATH = '../runs/yolo-train/detect/train4/weights/best.pt'
model = YOLO(MODEL_PATH)

def detect_animals_video(video_file):
    # temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    # temp_input.write(video_file.read())
    # temp_input.close()

    cap = cv2.VideoCapture(video_file)
    frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)

    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    out = cv2.VideoWriter(temp_output.name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, conf=0.25)
        plotted = results[0].plot()
        out.write(plotted)

    cap.release()
    out.release()

    return temp_output.name

with gr.Blocks() as demo:
    gr.Markdown("# 🐾 Wildlife Detection with YOLOv8")
    gr.Markdown("Upload an image or video to detect animals using your trained model.")

    # with gr.Tab("Detect from Image"):
    #     image_input = gr.Image(type="pil")
    #     image_output = gr.Image()
    #     gr.Button("Detect").click(fn=detect_animals_image, inputs=image_input, outputs=image_output)

    with gr.Tab("Detect from Video"):
        video_input = gr.Video()
        video_output = gr.Video(interactive=False)
        gr.Button("Detect").click(fn=detect_animals_video, inputs=video_input, outputs=video_output)

# 🚀 Launch the app
demo.launch()