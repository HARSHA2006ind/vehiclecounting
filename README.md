# 🚗 AI-Based Vehicle Detection, Tracking and Counting System

## 📌 Project Overview
This project is an AI-powered traffic analysis system built using **Python**, **OpenCV**, and **YOLOv3**. The system detects vehicles from a traffic video, classifies them into different categories, tracks them using a simple Euclidean distance tracker, and provides approximate vehicle counts.

The project demonstrates the application of **Computer Vision** and **Deep Learning** in intelligent transportation systems.

---

## ✨ Features

- 🚘 Real-time vehicle detection
- 🏍️ Vehicle classification (Car, Motorcycle, Bus, Truck)
- 📦 Bounding box visualization
- 🔢 Vehicle ID tracking
- 📊 Approximate vehicle counting
- 📄 CSV report generation
- 🎥 Supports custom traffic videos

---

## 🛠️ Technologies Used

- Python 3.x
- OpenCV
- NumPy
- YOLOv3 (Darknet)
- Computer Vision
- Deep Learning

---

## 📂 Project Structure

```
Vehicle-Detection-Project/
│── p1.py                 # Main program
│── tracker.py            # Euclidean distance tracker
│── coco.names            # COCO class labels
│── yolov3-320.cfg        # YOLOv3 configuration
│── yolov3.weights        # Pre-trained YOLO weights
│── video.mp4             # Input traffic video
│── vehicle_count.csv     # Output vehicle count report
│── README.md             # Project documentation
```

---

## ⚙️ Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/vehicle-detection-project.git
cd vehicle-detection-project
```

2. Install the required libraries:

```bash
pip install opencv-python numpy
```

3. Download the YOLO files:
   - `yolov3.weights`
   - `yolov3-320.cfg`
   - `coco.names`

4. Place a traffic video in the project folder and name it `video.mp4`.

---

## ▶️ How to Run

Open a terminal inside the project folder and execute:

```bash
python p1.py
```

The program will:
1. Load the YOLO model.
2. Read the traffic video.
3. Detect and classify vehicles.
4. Track detected vehicles.
5. Display the processed video with bounding boxes and IDs.
6. Save approximate vehicle counts to `vehicle_count.csv`.

---

## 🚦 Vehicle Classes Detected

| Class ID | Vehicle Type |
|----------|-------------|
| 2 | Car |
| 3 | Motorcycle |
| 5 | Bus |
| 7 | Truck |

---

## 📸 Sample Output

- Vehicle detection with bounding boxes.
- Vehicle IDs for tracking.
- Approximate count of Cars, Motorcycles, Buses, and Trucks.
- CSV file containing final vehicle statistics.

---

## 🔮 Future Improvements

- Improve tracking accuracy using DeepSORT or ByteTrack.
- Add real-time webcam/CCTV support.
- Integrate smart traffic signal management.
- Generate traffic analytics dashboards.
- Detect traffic congestion and accidents automatically.

---

## 🌍 Applications

- Smart traffic monitoring
- Highway surveillance
- Intelligent transportation systems
- Vehicle analytics
- AI-based road safety solutions

---

## 👨‍💻 Author

**Harshavardhan**  
B.Tech Artificial Intelligence and Machine Learning (AIML)  
Rajalakshmi Institute of Technology, Chennai

---

## 📜 License

This project is developed for educational and academic purposes.
