# live_mask_detector.py
import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import os
import time
from threading import Thread
import queue

# Page configuration
st.set_page_config(
    page_title="Live Mask Detection",
    page_icon="😷",
    layout="wide"
)

# Constants
MODEL_PATH = os.path.join("saved_models", "mask_detector.h5")
IMAGE_SIZE = (224, 224)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 1rem;
}
.status-running {
    background-color: #d4edda;
    padding: 1rem;
    border-radius: 10px;
    border-left: 5px solid #28a745;
    text-align: center;
}
.status-stopped {
    background-color: #fff3cd;
    padding: 1rem;
    border-radius: 10px;
    border-left: 5px solid #ffc107;
    text-align: center;
}
.controls {
    text-align: center;
    margin: 2rem 0;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🎥 Live Camera Mask Detection</h1>', unsafe_allow_html=True)

@st.cache_resource
def load_mask_model():
    """Load the trained mask detection model"""
    try:
        if not os.path.exists(MODEL_PATH):
            st.error("❌ Model file not found! Please ensure 'mask_detector.h5' exists in the 'saved_models' folder.")
            return None
        model = load_model(MODEL_PATH)
        st.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

def load_face_detector():
    """Load OpenCV face detector"""
    try:
        face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        face_detector = cv2.CascadeClassifier(face_cascade_path)
        return face_detector
    except Exception as e:
        st.error(f"❌ Error loading face detector: {str(e)}")
        return None

def preprocess_face(face):
    """Preprocess face for model prediction"""
    face = cv2.resize(face, IMAGE_SIZE)
    face = face.astype("float") / 255.0
    face = img_to_array(face)
    face = np.expand_dims(face, axis=0)
    return face

class CameraHandler:
    def __init__(self, model, face_detector):
        self.model = model
        self.face_detector = face_detector
        self.cap = None
        self.running = False
        self.frame_queue = queue.Queue(maxsize=2)
        
    def start_camera(self):
        """Start the camera"""
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                # Try different camera indices
                for i in range(1, 5):
                    self.cap = cv2.VideoCapture(i)
                    if self.cap.isOpened():
                        break
                        
            if not self.cap.isOpened():
                return False
                
            # Set camera properties for better performance
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            self.running = True
            return True
        except Exception as e:
            st.error(f"Error starting camera: {e}")
            return False
    
    def stop_camera(self):
        """Stop the camera"""
        self.running = False
        if self.cap:
            self.cap.release()
    
    def process_frame(self):
        """Process a single frame"""
        if not self.cap or not self.running:
            return None
            
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Detect faces and classify
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(60, 60)
        )
        
        labels_map = {0: "MASK", 1: "NO MASK"}
        
        for (x, y, w, h) in faces:
            # Extract face
            face = frame[y:y+h, x:x+w]
            face_input = preprocess_face(face)
            
            # Predict
            preds = self.model.predict(face_input, verbose=0)
            class_idx = np.argmax(preds, axis=1)[0]
            prob = preds[0][class_idx]
            label = labels_map[class_idx]
            
            # Color coding: Green for mask, Red for no mask
            color = (0, 255, 0) if class_idx == 0 else (0, 0, 255)
            
            # Draw rectangle and label
            text = f"{label}: {prob*100:.1f}%"
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        
        # Convert BGR to RGB for Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame_rgb

# Load models
model = load_mask_model()
face_detector = load_face_detector()

if model is None or face_detector is None:
    st.stop()

# Initialize camera handler
if 'camera_handler' not in st.session_state:
    st.session_state.camera_handler = CameraHandler(model, face_detector)

# Control buttons
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown('<div class="controls">', unsafe_allow_html=True)
    
    col_start, col_stop = st.columns(2)
    with col_start:
        start_button = st.button("🚀 Start Camera", type="primary", use_container_width=True)
    with col_stop:
        stop_button = st.button("🛑 Stop Camera", type="secondary", use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Initialize session state
if 'camera_active' not in st.session_state:
    st.session_state.camera_active = False

# Handle button clicks
if start_button:
    if st.session_state.camera_handler.start_camera():
        st.session_state.camera_active = True
        st.rerun()
    else:
        st.error("❌ Failed to start camera. Please check your webcam connection.")

if stop_button:
    st.session_state.camera_handler.stop_camera()
    st.session_state.camera_active = False
    st.rerun()

# Status display
if st.session_state.camera_active:
    st.markdown("""
    <div class="status-running">
    <h3>🟢 Camera Active - Live Detection Running</h3>
    <p>Position your face clearly in front of the camera. The system will automatically detect mask usage.</p>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="status-stopped">
    <h3>⚪ Camera Stopped</h3>
    <p>Click "Start Camera" to begin live mask detection.</p>
    </div>
    """, unsafe_allow_html=True)

# Video feed placeholder
video_placeholder = st.empty()

# Main camera loop
if st.session_state.camera_active:
    try:
        while st.session_state.camera_active:
            frame = st.session_state.camera_handler.process_frame()
            if frame is not None:
                video_placeholder.image(frame, channels="RGB", use_container_width=True)
            else:
                st.error("❌ Failed to read from camera")
                st.session_state.camera_active = False
                break
            
            # Small delay to prevent overwhelming the system
            time.sleep(0.03)  # ~30 FPS
            
    except Exception as e:
        st.error(f"❌ Camera error: {str(e)}")
        st.session_state.camera_active = False
        st.session_state.camera_handler.stop_camera()

# Instructions
st.markdown("---")
with st.expander("📋 Instructions & Setup"):
    st.markdown("""
    **How to Use:**
    1. Make sure your webcam is connected and working
    2. Click "Start Camera" to begin live detection
    3. Position your face clearly in front of the camera
    4. The system will show:
       - 🟢 Green box with "MASK" if wearing a mask
       - 🔴 Red box with "NO MASK" if not wearing a mask
    5. Click "Stop Camera" when finished
    
    **Setup Requirements:**
    ```bash
    pip install streamlit opencv-python tensorflow numpy
    ```
    
    **File Structure:**
    ```
    your_project/
    ├── live_mask_detector.py
    └── saved_models/
        └── mask_detector.h5
    ```
    
    **To Run:**
    ```bash
    streamlit run live_mask_detector.py
    ```
    
    **Troubleshooting:**
    - If camera doesn't start, try closing other applications using the webcam
    - Make sure you have granted camera permissions to your browser
    - The app will automatically try different camera indices if the default doesn't work
    """)

# Technical specs
st.markdown("---")
st.markdown("""
**🔧 Technical Specifications:**
- **Model**: MobileNetV2 with Transfer Learning
- **Face Detection**: OpenCV Haar Cascade
- **Input Resolution**: 224x224 pixels
- **Frame Rate**: ~30 FPS
- **Detection Classes**: Mask / No Mask
""")