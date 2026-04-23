# PlateTrack AI

AI-powered vehicle license plate detection and OCR system built with YOLO, OpenCV, EasyOCR, and Streamlit.

This project detects license plates from images and videos, draws bounding boxes with confidence scores, extracts plate text, and serves results through an interactive web interface.

## Author
- Dev Dharmesh Patel
- Email: devdpatel190905@gmail.com
- GitHub: https://github.com/devpatel0005

## Project Overview
PlateTrack AI is an end-to-end Automatic Number Plate Recognition (ANPR) project with two major parts:
1. Model development workflow in notebook for dataset preparation, training, and evaluation.
2. Production-style Streamlit UI for inference on uploaded media (images/videos).

## Key Features
- License plate detection using trained YOLO model (`models/best.pt`)
- Image inference with bounding box visualization
- Video inference frame-by-frame with bounding box overlays
- OCR extraction of detected plate text from image regions (EasyOCR)
- Browser-friendly video output conversion using FFmpeg
- Resume-ready Streamlit frontend with project info, stack, and use cases

## Tech Stack
- Python
- Streamlit
- Ultralytics YOLO
- OpenCV
- EasyOCR
- FFmpeg (via `imageio-ffmpeg`)
- NumPy, Pandas, Matplotlib

## Repository Structure
```text
Automatic-Car-Numberplate-Recognition-System/
|-- yolo_application.py
|-- requirements.txt
|-- README.md
|-- LICENSE
|-- models/
|   `-- best.pt
|-- notebooks/
|   `-- Car Numberplate Detection.ipynb
|-- output/
|-- temp/
|-- datasets/
`-- demo.mp4
```

## How It Works
### 1) Detection
- YOLO model predicts plate bounding boxes (`x1, y1, x2, y2`) and confidence.
- Boxes and confidence text are drawn on image/video frames.

### 2) OCR
- For each detected plate in images, ROI is cropped from the bounding box.
- EasyOCR reads text from ROI.
- Extracted text is shown in Streamlit.

### 3) Video Preview Conversion
- Processed video is converted to web-friendly MP4 (H.264 + yuv420p + faststart).
- This improves compatibility in browser playback.

## Model Development (Notebook)
The notebook `notebooks/Car Numberplate Detection.ipynb` includes:
- XML annotation parsing and dataframe creation
- Pascal VOC to YOLO coordinate conversion
- Train/validation/test split preparation
- YOLO training and metric plotting
- Image-level detection testing
- OCR integration experiments
- Video detection workflow

## Installation
From project root:

```bash
pip install -r requirements.txt
```

## Run the Application
```bash
streamlit run yolo_application.py
```

Then open the Streamlit URL shown in terminal and upload:
- Image: `.jpg`, `.jpeg`, `.png`, `.bmp`
- Video: `.mp4`, `.avi`, `.mov`, `.mkv`

## Example Output
- Input image/video with detected license plate boxes
- Confidence labels on each detection
- OCR text output for image-based detections
- Converted preview video in `output/` for browser playback

## Resume Highlights (You Can Reuse)
- Built an end-to-end ANPR pipeline using YOLO for object detection and EasyOCR for text recognition.
- Converted raw annotation formats into YOLO-ready datasets and trained a custom detection model.
- Developed a Streamlit-based inference app for image and video uploads with real-time visual outputs.
- Implemented robust media processing and browser-compatible video rendering using FFmpeg.

## License
This project is proprietary and licensed under an All Rights Reserved license.
See `LICENSE` for full terms.
