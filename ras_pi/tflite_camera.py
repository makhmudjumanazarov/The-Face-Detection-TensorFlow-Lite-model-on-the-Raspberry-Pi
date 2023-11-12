import streamlit as st
import cv2
import tensorflow as tf
import numpy as np
from PIL import Image
import json
from numpy import ndarray
from typing import List, Optional, Tuple, Union
import tempfile
import time
from core import *

st.write("""### Face Detection""")

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="yolov8.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Function to initialize video capture
cap  = cv2.VideoCapture()
cap.open("rtsp://admin:AEZAKMI12@192.168.0.161/:554/Streaming/channels/2/")


def predictVideo(cap, conf_threshold = 0.5, iou_thres = 0.1):
    if cap is not None:

        if cap.isOpened():
            st.write("Video Playback:")
            prev_time = 0
            curr_time = 0
            fps_out = st.empty()
            image_out = st.empty()

            (success, image) = cap.read()

            while success:

                prev_time = time.time()
                
                input_shape = (640, 640)  # Replace with the expected input shape of your YOLO model
                bgr, ratio, dwdh = letterbox(np.array(image), input_shape)
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                tensor = blob(rgb, return_seg=False)
                tensor = np.ascontiguousarray(tensor)
                
                interpreter.set_tensor(input_details[0]['index'], tensor)
                interpreter.invoke()
                # The function `get_tensor()` returns a copy of the tensor data.
                # Use `tensor()` in order to get a pointer to the tensor.
                result = interpreter.get_tensor(output_details[2]['index'])
                predictions = np.array(result).reshape((5, 8400))
                predictions = predictions.T 

                curr_time = time.time()
                fps = 1.0 / (curr_time - prev_time)
                fps_out.write(f"FPS:{fps}")

                # Filter out object confidence scores below threshold
                scores = np.max(predictions[:, 4:], axis=1)
                predictions = predictions[scores > conf_threshold, :]
                scores = scores[scores > conf_threshold] 
                class_ids = np.argmax(predictions[:, 4:], axis=1)

                # Get bounding boxes for each object
                boxes = predictions[:, :4]

                #rescale box
                input_shape = np.array([640, 640, 640, 640])
                boxes = np.divide(boxes, input_shape, dtype=np.float32)
                boxes *= np.array([640, 640, 640, 640])
                boxes = boxes.astype(np.int32)

                indices = nms(boxes, scores, iou_thres)
                image_draw = rgb.copy()

                for (bbox, score, label) in zip(xywh2xyxy(boxes[indices]), scores[indices], class_ids[indices]):
                    bbox = bbox.round().astype(np.int32).tolist()
                    cls_id = int(label)
                    color = (0,255,0)
                    cv2.rectangle(image_draw, tuple(bbox[:2]), tuple(bbox[2:]), color, 2)
                    cv2.putText(image_draw, f'face:{int(score*100)}',
                                (bbox[0], bbox[1] - 2),
                                cv2.FONT_HERSHEY_PLAIN,
                                1, [225, 255, 255],
                                thickness=1) 

                output_image = cv2.cvtColor(image_draw, cv2.COLOR_BGR2RGB)
                
                # Display the frame in Streamlit
                image_out.image(output_image, channels="BGR", use_column_width=True)
                # cv2.imwrite(f"./detected_images/rasm.png", output_image)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

                (success, image) = cap.read()
                    # Release everything after the job is finished
            cap.release()
            # out.release()
            cv2.destroyAllWindows()
        else:
            st.write("Error: Unable to open the video file.")
    else:
        st.write("Please upload a video file to display.")

def main():
    predictVideo(cap)

if __name__ == "__main__":
    main()
