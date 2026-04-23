import streamlit as st
import os
import subprocess
import imageio_ffmpeg as iio_ffmpeg
from ultralytics import YOLO

st.title("YOLO image and vedio Processing.")

# Allow users to upload images or videos
uploaded_file = st.file_uploader("Upload an image or video", type=["jpg", "jpeg", "png", "bmp", "mp4", "avi", "mov", "mkv"])

# Load the model with giving the weights in which we ran our model in the notebook

model=YOLO('D:\\vscode\\Automatic-Car-Numberplate-Recognition-System\\models\\best.pt')



def get_saved_media_path(source_path, results):
    """Return YOLO's saved output path for a source image/video."""
    if not results:
        return None

    save_dir = str(results[0].save_dir)
    source_name = os.path.basename(source_path)
    direct_match = os.path.join(save_dir, source_name)
    if os.path.exists(direct_match):
        return direct_match

    source_stem = os.path.splitext(source_name)[0]
    for saved_name in os.listdir(save_dir):
        if os.path.splitext(saved_name)[0] == source_stem:
            return os.path.join(save_dir, saved_name)

    return None


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



# Prediction function for the images will take the image , will make the predictions and save the image and returns the output folder 
def predict_and_save_image(path_test_car):
    results = model.predict(
        source=path_test_car,
        device='cpu',
        save=True,
        project='output',
        name='predictions',
        exist_ok=True,
    )
    result_path = get_saved_media_path(path_test_car, results)
    if result_path is None:
        st.error("YOLO could not locate the saved output image.")
    return result_path


# Now prediction for the vedio input 
# for the vedio we need to predict for every image in a veiod for every frame per second we need to predict on image in vedio
def predict_and_plot_video(video_path):
    """
    Predicts and saves the bounding boxes on the given test video using the trained YOLO model.

    Parameters:
    video_path (str): Path to the test video file.
    Returns:
    str: The path to the saved output video file.
    """
    results = model.predict(
        source=video_path,
        device='cpu',
        save=True,
        project='output',
        name='predictions',
        exist_ok=True,
    )
    result_path = get_saved_media_path(video_path, results)
    if result_path is None:
        st.error("YOLO could not locate the saved output video.")
        return None
    return result_path
    

# Error Handelling
def process_media(input_path):
    """
    Processes the uploaded media file (image or video) and returns the path to the saved output file.

    Parameters:
    input_path (str): Path to the input media file.
    Returns:
    str: The path to the saved output media file.
    """
    file_extension = os.path.splitext(input_path)[1].lower()
    if file_extension in ['.mp4', '.avi', '.mov', '.mkv']:
        return predict_and_plot_video(input_path)
    elif file_extension in ['.jpg', '.jpeg', '.png', '.bmp']:
        return predict_and_save_image(input_path)
    else:
        st.error(f"Unsupported file type: {file_extension}")
        return None




# here we are saving the uploaded image in the temp directory 
if uploaded_file is not None:
    os.makedirs("temp", exist_ok=True)
    input_path=f"temp/{uploaded_file.name}"
    with open(input_path,'wb') as f:
        f.write(uploaded_file.getbuffer())
    st.write("Processing....")

    result_path = process_media(input_path)
    if result_path:
        if input_path.endswith(('.mp4', '.avi', '.mov', '.mkv')):
            st.write(f"Saved output path: {result_path}")
            preview_video_path = create_web_preview_video(result_path)
            if preview_video_path:
                st.video(preview_video_path)
        else:
            st.image(result_path)
 
