Smart IoT Anomaly Detection System

Real-time IoT anomaly detection for temperature, humidity, and sound sensors.
Designed for predictive maintenance, early failure detection, and live monitoring.

Features

Real-time anomaly detection

Streaming data simulation

REST API for predictions (/predict)

Retrainable ML model (/retrain)

Docker-ready & monitoring-friendly

Project Structure
api_service.py       # REST API for live inference
train_model.py       # Train the initial model
retrain.py           # Retraining pipeline
simulate_stream.py   # Simulate live sensor data
utils.py             # Helper functions
data/                # Dataset
Dockerfile
requirements.txt
README.md

Installation
git clone https://github.com/Ahsanc369/IoT-Anomaly-Detection-System.git
cd IoT-Anomaly-Detection-System
pip install -r requirements.txt

Usage
Train the Model
python train_model.py

Start the API
python api_service.py

Simulate Streaming Data
python simulate_stream.py
