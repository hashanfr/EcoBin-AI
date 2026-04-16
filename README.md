## EcoBin – AI + IoT Powered Smart Waste Management System

An AI-powered smart waste management system that automatically classifies waste into biodegradable and non-biodegradable categories using computer vision. The system simulates smart disposal actions such as biogas conversion and waste compaction, along with a monitoring dashboard for smart city applications.

## Features
Live camera-based waste detection
Image upload support for classification
AI-based waste classification using deep learning
Automatic routing of waste:
Biodegradable → Biogas unit
Non-biodegradable → Compactor
Interactive Streamlit dashboard
Simulated bin level monitoring system
Waste processing statistics display
Auto-dump system simulation for smart management
## Tech Stack
Python
TensorFlow (MobileNetV2)
Streamlit
NumPy
Pillow
## Project Structure
ecobin/
│
├── dataset/
│   ├── train/
│   │   ├── biodegradable/
│   │   └── non_biodegradable/
│   │
│   └── val/
│       ├── biodegradable/
│       └── non_biodegradable/
│
├── train.py
├── app.py
├── model.h5
├── requirements.txt
└── README.md
## How to Run
git clone https://github.com/your-username/ecobin.git
cd ecobin
pip install -r requirements.txt
python train.py
streamlit run app.py
## How It Works
Capture or upload waste image
AI model classifies waste type
System decides processing path:
Biodegradable → Biogas production
Non-biodegradable → Compaction
Dashboard updates system status in real-time
## Output
Waste classification result (Biodegradable / Non-biodegradable)
Confidence score
Action routing (Biogas / Compactor)
Dashboard metrics:
Bin level (simulated)
Waste processed
Biogas generated
Collection status
## Future Improvements
Real-time object detection with bounding boxes
IoT-based actual bin level monitoring using sensors
Mobile application deployment
Cloud-based dashboard for municipalities
Smart city integration at scale
## Author

Hashanthra K
