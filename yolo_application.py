import streamlit as st
import os
import cv2
import subprocess
import tempfile
import imageio_ffmpeg as iio_ffmpeg
import easyocr
from ultralytics import YOLO

st.set_page_config(page_title="PlateTrack AI", page_icon="AI", layout="wide")

st.title("PlateTrack AI")
st.subheader("AI-powered vehicle license plate detection and OCR from images and videos")

st.sidebar.title("Project Info")
st.sidebar.markdown("**Author:** Dev Dharmesh Patel")
st.sidebar.markdown("**Mail:** devdpatel0005@gmail.com")
st.sidebar.markdown("**GitHub:** https://github.com/devpatel0005")
st.sidebar.markdown("---")
st.sidebar.markdown("**Tech Stack**")
st.sidebar.markdown("- Python")
st.sidebar.markdown("- Streamlit")
st.sidebar.markdown("- YOLO (Ultralytics)")
st.sidebar.markdown("- OpenCV")
st.sidebar.markdown("- EasyOCR")
st.sidebar.markdown("- FFmpeg")
st.sidebar.markdown("---")
st.sidebar.markdown("**License**")
st.sidebar.markdown("All Rights Reserved")
st.sidebar.markdown("No reuse or redistribution without written permission.")
st.sidebar.markdown("Contact: devdpatel0005@gmail.com")
st.sidebar.markdown("---")
st.sidebar.markdown("**Use Cases**")
st.sidebar.markdown("- Parking access automation")
st.sidebar.markdown("- Traffic monitoring")
st.sidebar.markdown("- Fleet entry logs")

st.divider()

# Allow users to upload images or videos
uploaded_file = st.file_uploader("Upload an image or video to detect number plates and extract text.", type=["jpg", "jpeg", "png", "bmp", "mp4", "avi", "mov", "mkv"])

# Load the model with giving the weights in which we ran our model in the notebook
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best.pt")
model = YOLO(MODEL_PATH)

APP_TEMP_DIR = os.path.join(tempfile.gettempdir(), "platetrack_ai")
INPUT_DIR = os.path.join(APP_TEMP_DIR, "input")
OUTPUT_DIR = os.path.join(APP_TEMP_DIR, "output")
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


@st.cache_resource
def get_ocr_reader():
    return easyocr.Reader(['en'], gpu=False)


def create_web_preview_video(input_video_path):
    """Create a browser-friendly MP4 preview using FFmpeg (H.264 + yuv420p)."""
    preview_path = f"{os.path.splitext(input_video_path)[0]}_web.mp4"
    ffmpeg_exe = iio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg_exe,
        "-y",
        "-i",
        input_video_path,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-c:a",
        "aac",
        preview_path,
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        error_message = exc.stderr.strip() if exc.stderr else "Unknown FFmpeg error"
        st.error(f"FFmpeg conversion failed: {error_message}")
        return None

    if not os.path.exists(preview_path):
        st.error("FFmpeg did not create the preview video.")
        return None

    return preview_path

def predict_and_save_image(path_test_car, output_image_path):
    results= model.predict(path_test_car,device='cpu')
    image=cv2.imread(path_test_car)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB) # Conver the image into the RGB format deafault format is BGR
    detected_texts = []
    ocr_reader = get_ocr_reader()
    
    # Now we need to format the predicts stored inside the result
    for result in results:
        for box in result.boxes:
            x1,y1,x2,y2=map(int,box.xyxy[0]) #This is the box in XYXY format: [left top coordinates x1,y1, right bottom coordinates x2,y2]. inside the results boxes
            confidence=box.conf[0]
            #now we will pass this coordinates and plot the box 
            cv2.rectangle(image,(x1,y1),(x2,y2),(0,255,0),2)
            # now we will write confidence score on the image with custom font
            cv2.putText(image, f'{confidence*100:.2f}%', (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            roi = image[y1:y2, x1:x2]
            if roi.size > 0:
                text_parts = ocr_reader.readtext(roi, detail=0)
                if text_parts:
                    text = " ".join(text_parts).strip()
                    if text:
                        detected_texts.append(text)
    
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # convert to default format
    cv2.imwrite(output_image_path, image)
    st.session_state['ocr_texts'] = detected_texts
    return output_image_path


# Now prediction for the vedio input 
# for the vedio we need to predict for every image in a veiod for every frame per second we need to predict on image in vedio
def predict_and_plot_video(video_path,output_path):
    """
    Predicts and saves the bounding boxes on the given test video using the trained YOLO model.

    Parameters:
    video_path (str): Path to the test video file.
    output_path (str): Path to save the output video file.

    Returns:
    str: The path to the saved output video file.
    """
    cap=cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error(f"Error opening video file: {video_path}")   
        return None
    frame_width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps=int(cap.get(cv2.CAP_PROP_FPS))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model.predict(rgb_frame, device='cpu')
        # Now below is the same logic as we used while predicting in the image, here will be frame instead of image  
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'{confidence*100:.2f}%', (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        out.write(frame)
    cap.release()
    out.release()
    return output_path
    

# Error Handelling
def process_media(input_path, output_path):
    """
    Processes the uploaded media file (image or video) and returns the path to the saved output file.

    Parameters:
    input_path (str): Path to the input media file.
    output_path (str): Path to save the output media file.

    Returns:
    str: The path to the saved output media file.
    """
    file_extension = os.path.splitext(input_path)[1].lower()
    if file_extension in ['.mp4', '.avi', '.mov', '.mkv']:
        st.session_state['ocr_texts'] = []
        return predict_and_plot_video(input_path, output_path)
    elif file_extension in ['.jpg', '.jpeg', '.png', '.bmp']:
        return predict_and_save_image(input_path, output_path)
    else:
        st.error(f"Unsupported file type: {file_extension}")
        return None




# here we are saving the uploaded image in the temp directory 
if uploaded_file is not None:
    safe_name = os.path.basename(uploaded_file.name)
    input_path = os.path.join(INPUT_DIR, safe_name)
    output_path = os.path.join(OUTPUT_DIR, safe_name)
    with open(input_path,'wb') as f:
        f.write(uploaded_file.getbuffer())
    preview_path = None
    with st.spinner("Processing..."):
        result_path = process_media(input_path, output_path)
        if result_path and input_path.endswith(('.mp4', '.avi', '.mov', '.mkv')):
            preview_path = create_web_preview_video(result_path)

    if result_path:
        if input_path.endswith(('.mp4', '.avi', '.mov', '.mkv')):
            left_spacer, video_col, right_spacer = st.columns([1, 2, 1])
            if preview_path:
                with video_col:
                    st.video(preview_path)
            else:
                with open(result_path, 'rb') as video_file:
                    video_bytes = video_file.read()
                with video_col:
                    st.video(video_bytes)
        else:
            st.image(result_path)
            ocr_texts = st.session_state.get('ocr_texts', [])
            if ocr_texts:
                st.write("Detected text from number plate:")
                for text in ocr_texts:
                    st.write(text)

st.markdown(
    """
    <style>
    .app-footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        text-align: center;
        padding: 8px 12px;
        background: rgba(0, 0, 0, 0.45);
        color: #c9ced6;
        font-size: 13px;
        z-index: 999;
        backdrop-filter: blur(2px);
    }
    .block-container {
        padding-bottom: 48px;
    }
    </style>
    <div class="app-footer">Copyright (c) 2026 Dev Dharmesh Patel. All Rights Reserved.</div>
    """,
    unsafe_allow_html=True,
)
 
