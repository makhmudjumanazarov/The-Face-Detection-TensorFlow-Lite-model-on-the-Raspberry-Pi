## The Face Detection TensorFlow Lite model on the Raspberry-Pi
## Important
I built a model for Face Detection using YOLOv8. First I exported the YOLOv8 model -> ONNX -> Tensorflow -> Tensorflow Lite. As a result, when i predicted the video through the Tensorflow Lite model, the FPS went up to (5~6) and deployed via streamlit on the Raspberry-Pi.
- Trained YOLOv8 model for Face Detection: <a href= "https://drive.google.com/file/d/11t3sReQt1xrl7n0Mqo5we865azARmGiT/view?usp=sharing"> face.pt </a>
- Pytorch -> ONNX -> Tensorflow -> Tensorflow Lite Conversion :  <a href= "https://github.com/makhmudjumanazarov/YOLOv8-convert-ONNX-Tensorflow-TFLite-and-ONNX-TensorRT"> repo </a>

### Steps to Use
<b>Step 1.</b> Clone a repo
<pre>
git clone https://github.com/makhmudjumanazarov/The-Face-Detection-TensorFlow-Lite-model-on-the-Raspberry-Pi.git
</pre> 
<b>Step 2.</b> Create a new virtual environment 
<pre>
python -m venv ras_tf_lite
</pre> 
<br/>
<b>Step 3.</b> Activate your virtual environment
<pre>
source ras_tf_lite/bin/activate # Linux/Debian
</pre>
<br/>
<b>Step 4.</b> Install dependencies and add virtual environment to the Python Kernel
<pre>
python -m pip install --upgrade pip
pip install -r requirements.txt 
pip install ipykernel
python -m ipykernel install --user --name=ras_tf_lite
</pre>
<br/>
<b>Step 5.</b> Fix rtsp link
<pre>
rtsp://{user}:{password}@{ip_address}/:{port}/Streaming/channels/2/
  
example: rtsp://admin:AEZAKMI12@192.168.0.161/:554/Streaming/channels/2/
</pre>

<b>Step 6.</b> Run streamlit on localhost by running the stream.py file via terminal command (You can select an optional port)
<pre>
streamlit run tflite_camera.py --server.port 8520
</pre>
