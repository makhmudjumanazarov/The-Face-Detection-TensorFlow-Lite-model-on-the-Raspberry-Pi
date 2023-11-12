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
python -m venv TensorRT
</pre> 
<br/>
<b>Step 3.</b> Activate your virtual environment
<pre>
source TensorRT/bin/activate # Linux
</pre>
<br/>
<b>Step 4.</b> Install dependencies and add virtual environment to the Python Kernel
<pre>
python -m pip install --upgrade pip
pip install -r requirements.txt 
pip install ipykernel
python -m ipykernel install --user --name=TensorRT
</pre>
<br/>
<b>Step 5.</b> Run streamlit on localhost by running the stream.py file via terminal command (You can select an optional port)
<pre>
streamlit run stream.py --server.port 8520
</pre>

<br/>
<b>Step 6.</b> Open another page in the terminal (it should be the same as the path above). 
<pre>
  - Sign up: https://ngrok.com/
  - Connect your account: 
                        1. ngrok config add-authtoken your token
                        2. ngrok http 8520     
                        
</pre>
<br/>



