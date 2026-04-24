# PlateTrack AI

An end-to-end, production-ready AI system for automatic vehicle license plate detection and text extraction, powered by YOLO, OpenCV, and EasyOCR.

PlateTrack AI takes you from model training to real-world deployment: it detects plates in images and videos, performs OCR with high accuracy, and delivers results through a deployed interactive web application.

## Deployed Project
- Status: Successfully deployed
- Live Demo: [Add your deployed project URL here](#)


## Project Overview
PlateTrack AI is a completed end-to-end Automatic Number Plate Recognition (ANPR) project with three major parts:
1. Model development workflow in notebook for dataset preparation, training, and evaluation.
2. Streamlit UI for inference on uploaded media (images/videos).
3. Deployment of the application for live web access.

## Key Features
- License plate detection using trained YOLO model (`models/best.pt`)
- Image inference with bounding box visualization
- Video inference frame-by-frame with bounding box overlays
- OCR extraction of detected plate text from image regions (EasyOCR)
- Browser-friendly video output conversion using FFmpeg
- Streamlit frontend with project info, stack, and use cases
- Deployed web application for real-world usage and demo access

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
|-- .gitignore
|-- yolo_application.py
|-- requirements.txt
|-- README.md
|-- LICENSE
|-- datasets/
|   |-- annotations/
|   |-- images/
|   `-- demo.mp4
|-- models/
|   `-- best.pt
|-- notebooks/
|   `-- Car Numberplate Detection.ipynb
|-- output/
|   |-- demo.mp4
|   |-- demo_web.mp4
|   `-- images.jpeg
|-- temp/
|   |-- demo.mp4
|   `-- images.jpeg
|-- test/
|   |-- demo.mp4
|   `-- images.jpeg
`-- yoloenv/
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

## Project Highlights 
- Built an end-to-end ANPR pipeline using YOLO for object detection and EasyOCR for text recognition.
- Converted raw annotation formats into YOLO-ready datasets and trained a custom detection model.
- Developed a Streamlit-based inference app for image and video uploads with real-time visual outputs.
- Implemented robust media processing and browser-compatible video rendering using FFmpeg.

## Future Improvements
In a production-scale next phase, I would integrate this deployed license plate detection system with CCTV streams using RTSP (Real-Time Streaming Protocol). Each frame would be processed in real time using YOLO for detection and OCR for text extraction. To optimize performance, I would add object tracking to avoid redundant OCR on the same vehicle across consecutive frames, then store detections in a database with timestamps and camera identifiers. The pipeline would be exposed through APIs and monitored through a dashboard for live events, search, and analytics.

## License
This project is proprietary and licensed under an All Rights Reserved license.
See `LICENSE` for full terms.

## Author
- Dev Dharmesh Patel
- Email: devdpatel190905@gmail.com
- GitHub: https://github.com/devpatel0005