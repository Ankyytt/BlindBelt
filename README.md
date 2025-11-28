# BlindBelt

Blind Assistance Smart Belt

A real-time safety and navigation support system designed for visually impaired individuals. The belt integrates multiple sensors and deep learning–based video analysis to detect obstacles, falls, or emergency situations and instantly notifies guardians via Telegram, Email, and local sound alerts. It operates on a multi-threaded, event-driven architecture to ensure fast and reliable performance.

Key Features

Real-Time Obstacle Detection using ultrasonic sensors

Fall Detection powered by MPU6050 motion analysis

GPS Live Location Tracking for emergency assistance

Deep Learning–based Visual Detection using MobileNet SSD

Multi-channel Alerts: Telegram, Email, and Buzzer/Alarm

Emergency Button for instant manual alerts

Parallel Sensor Fusion via multi-threading for zero-lag performance

🧠 System Architecture

All sensors and processing modules run concurrently using multi-threading

Event-driven responses trigger immediate action during threats

Sensor fusion combines motion, vision, and distance data for high reliability

📍 Core Functional Modules
Module	Function
Ultrasonic Sensors	Measure obstacle distance to avoid collisions
MPU6050	Detect sudden motion changes for fall prediction
GPS Module	Continuously updates user coordinates
Camera + DNN (MobileNet SSD)	Detects objects/environment hazards
Emergency Switch	Sends instant alert with location
Buzzer Alert	Local sound notification in critical events
🛠 Hardware & Interfaces
Component	Connection / Interface
Raspberry Pi 3 / 4	Central processing unit
Camera Module	CSI interface
MPU6050	I²C (GPIO 2 & 3)
GPS Module	UART / Serial
Ultrasonic Sensors	GPIO digital pins
Emergency Button	GPIO
Buzzer / Alarm	GPIO output

⚠ Ultrasonic sensor echo pin requires a 5V → 3.3V voltage divider to protect the Raspberry Pi.

💻 Software & Libraries
Tool / Library	Purpose
Python	Coding framework and threading
OpenCV	Video capture and processing
MobileNet SSD (Caffe)	Object detection
smtplib + STARTTLS	Email alerts
python-requests	Telegram messaging
RPI.GPIO & smbus	Hardware interface
📦 Output & Alerts

Alerts are sent automatically when:
✔ A fall is detected
✔ An obstacle is very close
✔ Hazard detected in camera view
✔ Emergency button pressed

Alert Includes:

Type of emergency

Live GPS location link

Snapshot (optional with camera)

Timestamp

🎯 Applications

Blind/visually impaired mobility assistance

Senior citizen fall monitoring

Personal safety wearable

Smart emergency response device

🔮 Future Improvements

Voice assistant guidance

IoT dashboard with cloud monitoring

Indoor navigation using BLE/Ultrasonic triangulation

Battery optimization and power-bank integration
