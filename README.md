# 🦯 Blind Assistance Smart Belt
A real-time wearable solution designed for visually impaired individuals, integrating multiple sensors and AI-based vision to detect obstacles, monitor falls, and send emergency alerts with live location tracking. Built using **Raspberry Pi**, **Python**, and **MobileNet SSD**.

---

## 🚀 Key Features
- Real-time **obstacle detection** using Ultrasonic sensors
- **Fall detection** using MPU6050 motion sensor
- **GPS-based live location tracking**
- **Object recognition** using MobileNet SSD (Deep Learning)
- **Multi-channel alerts**: Telegram, Email & Local Buzzer
- **Emergency button** for immediate distress alerts
- **Multi-threaded event-driven** architecture for fast response

---

## 🧠 System Architecture
All components run concurrently using a multi-threaded model:
- Parallel data collection from sensors
- Instant event-triggered responses
- Sensor fusion improves reliability & accuracy

---

## 📍 Functional Modules
| Module | Purpose |
|--------|---------|
| Ultrasonic Sensors | Detect obstacle distance |
| MPU6050 | Fall detection based on motion behavior |
| GPS Module | Continuous location streaming |
| Camera + MobileNet SSD | Real-time hazard classification |
| Emergency Button | Manual quick alert trigger |
| Buzzer | Local alert notification |

---

## 🛠 Hardware Components
| Component | Interface |
|-----------|-----------|
| Raspberry Pi 3 / 4 | Core computation |
| Camera Module | CSI Port |
| MPU6050 | I²C (GPIO 2,3) |
| GPS Module | UART Serial |
| Ultrasonic Sensors | GPIO |
| Emergency Button | GPIO |
| Buzzer / Alarm | GPIO |

⚠ **Important:** HC-SR04 Echo Pin must be reduced from **5V → 3.3V** using a voltage divider.

---

## 💻 Software & Tools
| Tool / Library | Description |
|----------------|-------------|
| Python | Development Platform |
| OpenCV | Video Processing |
| MobileNet SSD (Caffe) | Object Detection Model |
| python-requests | Telegram Alerts |
| smtplib + STARTTLS | Email Notifications |
| RPI.GPIO / smbus | Sensor Communication |

---

## 📦 Output & Alerts
Triggered automatically when:
- A fall is detected
- Obstacle distance is dangerously low
- Hazard detected via camera
- Emergency button is pressed

**Each alert includes:**
- Type of incident
- Live GPS location link
- Timestamp
- Optional snapshot (camera)

---

## 🎯 Use Cases
- Visually impaired navigation assistance
- Senior citizen fall monitoring
- Smart emergency wearable

---
