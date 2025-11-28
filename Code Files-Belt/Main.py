#!/usr/bin/env python3
# pi_monitor.py  - Fixed GPS version with proper error handling + SMTP email alerts

import os
import time
import sys
import threading
import webbrowser
from pathlib import Path

import tkinter as tk
from tkinter import ttk

import cv2
import numpy as np
import requests

# Pillow optional (used for Tk preview)
try:
    from PIL import Image, ImageTk
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False
    print("PIL not available - camera preview disabled")

# Guarded RPi / smbus imports (gives readable errors when run off-Pi)
try:
    import RPi.GPIO as GPIO
    ON_PI = True
except Exception:
    ON_PI = False
    class _FakeGPIO:
        BCM = IN = OUT = PUD_UP = LOW = HIGH = None
        def setmode(self, *a, **k): pass
        def setup(self, *a, **k): pass
        def input(self, *a, **k): return 1
        def output(self, *a, **k): pass
        def cleanup(self, *a, **k): pass
    GPIO = _FakeGPIO()
    print("Warning: RPi.GPIO not available. Running in non-hardware (test) mode.")

try:
    import smbus
    SMBUS_AVAILABLE = True
except Exception:
    SMBUS_AVAILABLE = False
    print("Warning: smbus not available. MPU6050 functions will be disabled.")

# -----------------------------
# CONSTANTS / CONFIG
# -----------------------------
EMERGENCY_PIN = 17

ULTRASONICS = {
    "Left": {"TRIG": 23, "ECHO": 24, "wav": "left.wav"},
    "Right": {"TRIG": 27, "ECHO": 22, "wav": "right.wav"},
    "Back": {"TRIG": 6, "ECHO": 5, "wav": "back.wav"}
}

ULTRA_THRESHOLD = 20        # cm
ULTRA_COOLDOWN = 10        # seconds
last_ultra_alert_time = {k: 0 for k in ULTRASONICS.keys()}

TELEGRAM_BOT_TOKEN = "8592182368:AAFLgm0j5ObV2d28LMSkmkxIS_BnlJWT484"
TELEGRAM_CHAT_ID = "5610685031"
TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

# -----------------------------
# SMTP EMAIL SETTINGS (PLACEHOLDERS - replace with your values)
# -----------------------------
SMTP_SERVER = "smtp.gmail.com"     # e.g. smtp.gmail.com
SMTP_PORT = 587                    # e.g. 587 for STARTTLS
SMTP_USER = "ankit22csu216@ncuindia.edu" # your smtp username
SMTP_PASS = "cmditqeilnmmtote"    # your app password or SMTP password
ALERT_EMAIL = "ankit22csu216@ncuindia.edu"  # recipient address

STATIC_LAT = 28.6139
STATIC_LON = 77.2090

SOUNDS_DIR = Path("sounds")
SOUNDS_DIR.mkdir(parents=True, exist_ok=True)

LAST_EMERGENCY_TIME = 0
EMERGENCY_COOLDOWN = 10

# DNN model files (ensure they exist or skip model)
PROTOTXT = "MobileNetSSD_deploy.prototxt.txt"
MODEL = "MobileNetSSD_deploy.caffemodel"

# -----------------------------
# GPS Reader (threaded) - FIXED VERSION
# -----------------------------
import serial

class GPSReader:
    def __init__(self):
        self.ser = None
        self.lat_in_degrees = STATIC_LAT
        self.lon_in_degrees = STATIC_LON
        self.gps_available = False
        self.running = False
        self.gps_thread = None
        self.last_gps_data_time = 0
        self.gps_data_count = 0
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        self.initialize_gps()

    def initialize_gps(self):
        """Initialize GPS with proper error handling"""
        # Common GPS configurations for Raspberry Pi
        gps_configs = [
            {"port": "/dev/ttyAMA0", "baudrate": 9600},  # GPIO serial
            {"port": "/dev/serial0", "baudrate": 9600},  # Alias for GPIO serial
            {"port": "/dev/ttyS0", "baudrate": 9600},    # Mini UART
            {"port": "/dev/ttyUSB0", "baudrate": 9600},  # USB GPS
        ]
        
        for config in gps_configs:
            try:
                print(f"Attempting to initialize GPS on {config['port']}...")
                self.ser = serial.Serial(
                    port=config['port'],
                    baudrate=config['baudrate'],
                    timeout=2.0,
                    bytesize=serial.EIGHTBITS,
                    parity=serial.PARITY_NONE,
                    stopbits=serial.STOPBITS_ONE,
                    xonxoff=False,
                    rtscts=False,
                    dsrdtr=False
                )
                
                # Test if we can read from the port
                if self.ser.is_open:
                    # Clear any existing data in buffer
                    self.ser.reset_input_buffer()
                    self.ser.reset_output_buffer()
                    
                    # Try to read a line to verify GPS is working
                    test_line = self.ser.readline().decode('ascii', errors='ignore')
                    if test_line:
                        print(f"GPS successfully initialized on {config['port']}")
                        print(f"Test data: {test_line.strip()}")
                        self.gps_available = True
                        break
                    else:
                        print(f"GPS on {config['port']} opened but no data received")
                        self.ser.close()
                else:
                    print(f"Failed to open {config['port']}")
                    
            except Exception as e:
                print(f"GPS initialization failed on {config['port']}: {e}")
                if self.ser and self.ser.is_open:
                    self.ser.close()
                self.ser = None
                
        if not self.gps_available:
            print("GPS initialization failed on all ports. Using static coordinates.")
            print("To enable GPS, make sure:")
            print("1. GPS module is properly connected")
            print("2. Serial is enabled in raspi-config")
            print("3. No other services are using the serial port (like console)")
            return
            
        self.running = True
        self.reconnect_attempts = 0
        self.gps_thread = threading.Thread(target=self._gps_loop, daemon=True)
        self.gps_thread.start()
        print("GPS thread started successfully")

    def _convert_to_degrees(self, raw_value):
        """Convert NMEA coordinates to decimal degrees"""
        try:
            if not raw_value or raw_value == '':
                return 0.0
            decimal_value = float(raw_value) / 100.0
            degrees = int(decimal_value)
            minutes = decimal_value - degrees
            position = degrees + (minutes / 0.6)
            return round(position, 6)
        except Exception as e:
            print(f"GPS degree conversion error for '{raw_value}': {e}")
            return 0.0

    def _parse_gps_data(self, received_data):
        """Parse NMEA GPS data with robust error handling"""
        try:
            if "$GPGGA" in received_data:
                # Log raw data for debugging (first few characters only)
                if self.gps_data_count < 10:  # Only log first 10 messages
                    print(f"GPS Raw: {received_data[:80]}...")
                
                # Parse GPGGA sentence
                parts = received_data.split("$GPGGA,", 1)[1].split(',')
                
                if len(parts) >= 10:
                    nmea_lat = parts[1]  # Latitude
                    nmea_lat_dir = parts[2]  # N/S
                    nmea_lon = parts[3]  # Longitude  
                    nmea_lon_dir = parts[4]  # E/W
                    fix_quality = parts[5]  # GPS fix quality (0=no fix, 1=GPS, 2=DGPS)
                    satellites = parts[6]  # Number of satellites
                    
                    # Check if we have a valid GPS fix
                    if fix_quality and fix_quality.isdigit() and int(fix_quality) > 0:
                        if nmea_lat and nmea_lon and len(nmea_lat) > 0 and len(nmea_lon) > 0:
                            lat = self._convert_to_degrees(nmea_lat)
                            lon = self._convert_to_degrees(nmea_lon)
                            
                            # Apply direction (N/S, E/W)
                            if nmea_lat_dir == 'S':
                                lat = -lat
                            if nmea_lon_dir == 'W':
                                lon = -lon
                                
                            # Validate coordinates (rough bounds)
                            if -90 <= lat <= 90 and -180 <= lon <= 180:
                                print(f"GPS Fix: Lat={lat:.6f}, Lon={lon:.6f}, Satellites={satellites}, Quality={fix_quality}")
                                return lat, lon
                            else:
                                print(f"GPS: Invalid coordinates: lat={lat}, lon={lon}")
                        else:
                            print("GPS: Empty latitude/longitude data")
                    else:
                        if self.gps_data_count % 10 == 0:  # Don't spam
                            print(f"GPS: Waiting for fix (quality: {fix_quality}, satellites: {satellites})")
                else:
                    print(f"GPS: Incomplete GPGGA sentence ({len(parts)} parts)")
        except Exception as e:
            print(f"GPS parsing error: {e}")
        return None, None

    def _gps_loop(self):
        """Main GPS reading loop with proper error recovery"""
        consecutive_errors = 0
        max_consecutive_errors = 5
        
        while self.running:
            try:
                # Check if serial port is available and open
                if not self.ser or not self.ser.is_open:
                    if self.reconnect_attempts < self.max_reconnect_attempts:
                        print("GPS port closed, attempting to reconnect...")
                        self.reconnect_attempts += 1
                        self.initialize_gps()
                        time.sleep(2)
                        continue
                    else:
                        print("Max GPS reconnect attempts reached. Giving up.")
                        break
                
                # Read data from GPS
                if self.ser.in_waiting > 0:
                    line = self.ser.readline().decode('ascii', errors='ignore').strip()
                    if line:
                        consecutive_errors = 0  # Reset error counter on successful read
                        self.gps_data_count += 1
                        lat, lon = self._parse_gps_data(line)
                        if lat is not None and lon is not None:
                            self.lat_in_degrees = lat
                            self.lon_in_degrees = lon
                            self.last_gps_data_time = time.time()
                            if self.gps_data_count <= 10 or self.gps_data_count % 50 == 0:
                                print(f"GPS Update #{self.gps_data_count}: {lat:.6f}, {lon:.6f}")
                else:
                    # No data available, just wait
                    time.sleep(0.1)
                    
            except serial.SerialException as e:
                consecutive_errors += 1
                print(f"GPS Serial error #{consecutive_errors}: {e}")
                
                if consecutive_errors >= max_consecutive_errors:
                    print("Too many serial errors, attempting to reset GPS connection...")
                    try:
                        if self.ser and self.ser.is_open:
                            self.ser.close()
                    except:
                        pass
                    self.ser = None
                    consecutive_errors = 0
                    time.sleep(2)
                    
            except Exception as e:
                consecutive_errors += 1
                print(f"GPS Unexpected error #{consecutive_errors}: {e}")
                
                if consecutive_errors >= max_consecutive_errors:
                    print("Too many unexpected errors in GPS loop")
                    break
                    
                time.sleep(1)

        print("GPS thread stopped")

    def get_coordinates(self):
        """Get current coordinates (real GPS or fallback to static)"""
        return self.lat_in_degrees, self.lon_in_degrees

    def stop(self):
        """Stop GPS thread and close serial port"""
        self.running = False
        try:
            if self.ser and self.ser.is_open:
                self.ser.close()
                print("GPS serial port closed")
        except Exception as e:
            print(f"GPS stop error: {e}")

gps_reader = GPSReader()

# ... [REST OF THE CODE REMAINS THE SAME AS YOUR LAST WORKING VERSION] ...

# -----------------------------
# Telegram helper
# -----------------------------
def send_telegram(msg):
    try:
        # Remove emojis from messages (keeps simple plain text)
        msg = msg.replace("ALERT", "ALERT:").replace("EMERGENCY", "EMERGENCY:").replace("ALERT", "ALERT:")
        data = {"chat_id": TELEGRAM_CHAT_ID, "text": msg}
        requests.post(TELEGRAM_API_URL, data=data, timeout=3)
        print(f"Telegram sent: {msg}")
    except Exception as e:
        print(f"Telegram send failed: {e}")

# -----------------------------
# EMAIL HELPER (SMTP)
# -----------------------------
import smtplib
from email.mime.text import MIMEText

def send_email(subject, body):
    try:
        msg = MIMEText(body, "plain")
        msg["Subject"] = subject
        msg["From"] = SMTP_USER
        msg["To"] = ALERT_EMAIL

        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=10)
        server.ehlo()
        # Use STARTTLS if port is 587 (typical). If you use SSL (465), you'd use smtplib.SMTP_SSL(...)
        try:
            server.starttls()
            server.ehlo()
        except Exception:
            # some servers may not support starttls - ignore and proceed
            pass

        server.login(SMTP_USER, SMTP_PASS)
        server.sendmail(SMTP_USER, [ALERT_EMAIL], msg.as_string())
        server.quit()

        print(f"EMAIL SENT: {subject} -> {ALERT_EMAIL}")
    except Exception as e:
        print(f"Email send failed: {e}")

# -----------------------------
# GPIO setup (safe)
# -----------------------------
if ON_PI:
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(EMERGENCY_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    for s in ULTRASONICS.values():
        GPIO.setup(s["TRIG"], GPIO.OUT)
        GPIO.setup(s["ECHO"], GPIO.IN)

# -----------------------------
# MPU6050 (smbus) functions
# -----------------------------
MPU_ADDR = 0x68
mpu_available = False
if SMBUS_AVAILABLE:
    try:
        bus = smbus.SMBus(1)
        bus.write_byte_data(MPU_ADDR, 0x6B, 0)   # wake up
        mpu_available = True
        print("MPU6050 initialized")
    except Exception as e:
        print("MPU6050 not available:", e)
        mpu_available = False

def read_mpu6050():
    if not mpu_available:
        return 0.0, 0.0, 0.0, 1.0
    def rw(reg):
        h = bus.read_byte_data(MPU_ADDR, reg)
        l = bus.read_byte_data(MPU_ADDR, reg+1)
        v = (h << 8) + l
        if v >= 0x8000:
            return -((65535 - v) + 1)
        return v
    try:
        x = rw(0x3B)/16384.0
        y = rw(0x3D)/16384.0
        z = rw(0x3F)/16384.0
        mag = round((x*x + y*y + z*z) ** 0.5, 3)
        return round(x,2), round(y,2), round(z,2), mag
    except Exception as e:
        print("MPU read error:", e)
        return 0.0,0.0,0.0,1.0

# -----------------------------
# Fall detection params
# -----------------------------
FALL_THRESHOLD_LOW = 0.5
FALL_THRESHOLD_HIGH = 2.0
last_fall_alert = 0
FALL_COOLDOWN = 30

def get_map_link(lat=None, lon=None):
    if lat is None or lon is None:
        lat, lon = gps_reader.get_coordinates()
    return f"https://www.google.com/maps?q={lat},{lon}"

def check_fall(mag):
    global last_fall_alert
    now = time.time()
    if (mag < FALL_THRESHOLD_LOW or mag > FALL_THRESHOLD_HIGH) and (now - last_fall_alert > FALL_COOLDOWN):
        lat, lon = gps_reader.get_coordinates()
        msg = f"ALERT: Fall Detected! Mag={mag}\nLocation: {get_map_link(lat, lon)}"

        # Telegram
        send_telegram(msg)
        # Email
        send_email("Fall Detected!", msg)

        last_fall_alert = now
        print(f"Fall detected! Magnitude: {mag}")

# -----------------------------
# Ultrasonic (with timeouts)
# -----------------------------
def read_ultrasonic(name, timeout=0.03):
    """Return distance in cm or 999.0 on timeout/invalid"""
    if not ON_PI:
        return 999.0
    s = ULTRASONICS[name]
    try:
        GPIO.output(s["TRIG"], False)
        time.sleep(0.0001)
        GPIO.output(s["TRIG"], True)
        time.sleep(0.00001)
        GPIO.output(s["TRIG"], False)

        start_time = time.time()
        start = None
        # wait for echo to go HIGH
        while GPIO.input(s["ECHO"]) == 0:
            if time.time() - start_time > timeout:
                return 999.0
            start = time.time()

        # wait for echo to go LOW
        stop_time = time.time()
        while GPIO.input(s["ECHO"]) == 1:
            if time.time() - stop_time > timeout:
                return 999.0
            stop = time.time()

        if start is None or stop is None:
            return 999.0

        elapsed = stop - start
        dist = elapsed * 17150  # cm
        if dist <= 0 or dist > 10000:
            return 999.0
        return round(dist, 2)
    except Exception as e:
        print("Ultrasonic read error:", e)
        return 999.0

def read_emergency():
    if not ON_PI:
        return False
    try:
        # configured with pull-up; pressed == LOW
        return GPIO.input(EMERGENCY_PIN) == GPIO.LOW
    except Exception:
        return False

# -----------------------------
# Object detection DNN (optional)
# -----------------------------
CLASSES = ["background","aeroplane","bicycle","bird","boat","bottle","bus",
           "car","cat","chair","cow","diningtable","dog","horse","motorbike",
           "person","pottedplant","sheep","sofa","train","tvmonitor"]

DANGEROUS = {"car","dog","train"}  # use set and lower-case matching
net = None
if os.path.exists(PROTOTXT) and os.path.exists(MODEL):
    try:
        net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
        print("DNN loaded successfully")
    except Exception as e:
        print("Failed to load DNN:", e)
        net = None
else:
    print("DNN files not found; continuing without object detection.")
    print(f"Looking for: {PROTOTXT} and {MODEL}")

def play_sound_file(path: Path):
    try:
        if path.exists():
            # use aplay (non-blocking & quiet)
            os.system(f"aplay -q {str(path)} &")
    except Exception as e:
        print("Play sound error:", e)

# -----------------------------
# GUI Class
# -----------------------------
class GUI:
    def __init__(self, root):
        self.root = root
        root.title("Pi3 Object & Hardware Monitor")
        root.geometry("980x640")
        root.protocol("WM_DELETE_WINDOW", self.on_closing)

        style = ttk.Style()
        style.configure("Header.TLabel", font=("Arial", 16, "bold"))
        style.configure("Info.TLabel", font=("Arial", 12))

        top = ttk.Frame(root, padding=8)
        top.pack(fill="x")

        ttk.Label(top, text="Pi3 Object & Hardware Monitor", style="Header.TLabel").pack(side="left")

        ctr = ttk.Frame(top)
        ctr.pack(side="right")
        self.start_cam_btn = ttk.Button(ctr, text="Start Camera", command=self.start_camera)
        self.start_cam_btn.pack(side="left", padx=4)
        self.stop_cam_btn = ttk.Button(ctr, text="Stop Camera", command=self.stop_camera)
        self.stop_cam_btn.pack(side="left", padx=4)
        self.stop_cam_btn.state(["disabled"])
        self.gps_status_btn = ttk.Button(ctr, text="GPS Status", command=self.show_gps_status)
        self.gps_status_btn.pack(side="left", padx=4)

        main = ttk.Frame(root, padding=8)
        main.pack(fill="both", expand=True)

        left = ttk.LabelFrame(main, text="Hardware Status", padding=12)
        left.pack(side="left", fill="y", padx=6)

        self.mpu_lbl = ttk.Label(left, text="MPU6050: --", style="Info.TLabel")
        self.mpu_lbl.pack(anchor="w", pady=5)
        self.fall_lbl = ttk.Label(left, text="Fall: --", style="Info.TLabel")
        self.fall_lbl.pack(anchor="w", pady=5)
        self.ultra_lbl = ttk.Label(left, text="Ultrasonic: --", style="Info.TLabel")
        self.ultra_lbl.pack(anchor="w", pady=5)
        self.em_lbl = ttk.Label(left, text="Emergency: --", style="Info.TLabel")
        self.em_lbl.pack(anchor="w", pady=5)

        gps_status_frame = ttk.Frame(left)
        gps_status_frame.pack(anchor="w", pady=5)
        ttk.Label(gps_status_frame, text="GPS Status:", style="Info.TLabel").pack(side="left")
        self.gps_status_lbl = ttk.Label(gps_status_frame,
                                        text="Active" if gps_reader.gps_available else "Not Available",
                                        style="Info.TLabel",
                                        foreground="green" if gps_reader.gps_available else "red")
        self.gps_status_lbl.pack(side="left", padx=6)

        gpsf = ttk.Frame(left)
        gpsf.pack(anchor="w", pady=5)
        ttk.Label(gpsf, text="GPS Coordinates:", style="Info.TLabel").pack(side="left")
        self.gps_lbl = ttk.Label(gpsf, text="--,--", style="Info.TLabel")
        self.gps_lbl.pack(side="left", padx=6)

        ttk.Button(left, text="Open Map", command=lambda: webbrowser.open(get_map_link())).pack(pady=6)

        right = ttk.LabelFrame(main, text="Camera & Logs", padding=12)
        right.pack(side="left", fill="both", expand=True)

        # Camera status in GUI (no preview frame needed since we'll use external window)
        self.camera_status_label = ttk.Label(right, text="Camera: Not Started", style="Info.TLabel")
        self.camera_status_label.pack(pady=5)
        
        self.log_box = tk.Text(right, height=15, state="disabled")
        self.log_box.pack(fill="both", expand=True, pady=8)

        # Camera state
        self.camera_running = False
        self.cap = None
        self.external_window_open = False

        # start periodic hardware update
        self.update_hw()
        self.log("System initialized")

    def on_closing(self):
        self.log("Shutting down...")
        self.camera_running = False
        try:
            if self.cap:
                self.cap.release()
            # Close any open OpenCV windows
            cv2.destroyAllWindows()
        except Exception:
            pass
        gps_reader.stop()
        try:
            GPIO.cleanup()
        except Exception:
            pass
        self.root.destroy()

    def log(self, msg):
        t = time.strftime("%H:%M:%S")
        self.log_box.config(state="normal")
        self.log_box.insert("end", f"[{t}] {msg}\n")
        self.log_box.see("end")
        self.log_box.config(state="disabled")

    def show_gps_status(self):
        lat, lon = gps_reader.get_coordinates()
        status = "Active" if gps_reader.gps_available else "Not Available"
        self.log(f"GPS Status: {status}, Coordinates: {lat:.6f}, {lon:.6f}")
        self.log(f"GPS Data Count: {gps_reader.gps_data_count}")

    def start_camera(self):
        if self.camera_running:
            return
        self.camera_running = True
        t = threading.Thread(target=self.cam_loop, daemon=True)
        t.start()
        self.start_cam_btn.state(["disabled"])
        self.stop_cam_btn.state(["!disabled"])
        self.log("Camera started with external window.")

    def stop_camera(self):
        self.camera_running = False
        self.log("Camera stopping...")
        self.start_cam_btn.state(["!disabled"])
        self.stop_cam_btn.state(["disabled"])
        # Close external window
        cv2.destroyAllWindows()
        self.external_window_open = False

    def cam_loop(self):
        try:
            self.cap = cv2.VideoCapture(0)
            # Try to set resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            if not self.cap.isOpened():
                self.log("Camera not found.")
                self.camera_running = False
                return
                
            self.log("Camera opened successfully - displaying in external window")
        except Exception as e:
            self.log(f"Camera error: {e}")
            self.camera_running = False
            return

        frame_count = 0
        self.external_window_open = True
        
        while self.camera_running:
            ret, frame = self.cap.read()
            if not ret:
                self.log("Failed to read camera frame")
                time.sleep(0.1)
                continue

            frame_count += 1
            detection_count = 0

            # object detection
            if net is not None:
                try:
                    (h, w) = frame.shape[:2]
                    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
                    net.setInput(blob)
                    detections = net.forward()
                    
                    for i in range(0, detections.shape[2]):
                        confidence = detections[0, 0, i, 2]
                        if confidence > 0.3:
                            idx = int(detections[0, 0, i, 1])
                            if 0 <= idx < len(CLASSES):
                                label = CLASSES[idx]
                            else:
                                continue
                                
                            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                            (startX, startY, endX, endY) = box.astype("int")
                            
                            # Ensure bounding box is within frame dimensions
                            startX = max(0, startX)
                            startY = max(0, startY)
                            endX = min(w, endX)
                            endY = min(h, endY)
                            
                            # Draw bounding box and label
                            color = (0, 255, 0)  # Green
                            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                            y = startY - 15 if startY - 15 > 15 else startY + 15
                            cv2.putText(frame, f"{label}: {confidence:.2f}", (startX, y),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                            detection_count += 1
                            
                            # play sound if exists
                            wavpath = SOUNDS_DIR / f"{label}.wav"
                            play_sound_file(wavpath)

                            # dangerous detection notify (debounced)
                            now = time.time()
                            last = last_detection_alert.get(label, 0)
                            if label.lower() in DANGEROUS and now - last > 30:
                                lat, lon = gps_reader.get_coordinates()
                                msg = f"ALERT: Dangerous Object: {label} at Loc: {get_map_link(lat, lon)}"
                                send_telegram(msg)
                                send_email("Dangerous Object Detected", msg)
                                last_detection_alert[label] = now
                                self.log(f"Dangerous object detected: {label}")

                except Exception as e:
                    self.log(f"DNN error: {e}")

            # Display in external OpenCV window
            try:
                # Add frame counter and status to the frame
                cv2.putText(frame, f"Frame: {frame_count} | Objects: {detection_count}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Show the frame in external window
                cv2.imshow('Camera Feed - Pi Monitor', frame)
                
                # Check for key press to exit (optional)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):  # Press 'q' to quit camera
                    self.stop_camera()
                    
            except Exception as e:
                self.log(f"OpenCV display error: {e}")
                self.external_window_open = False

            # Update GUI status
            self.camera_status_label.config(text=f"Camera: Running - Frame {frame_count}, Detections: {detection_count}")

            # Log detection info occasionally
            if frame_count % 30 == 0:  # Every 30 frames
                self.log(f"Camera: Frame {frame_count}, Detections: {detection_count}")

            time.sleep(0.03)  # small sleep to reduce CPU

        # Cleanup when camera stops
        try:
            self.cap.release()
            cv2.destroyAllWindows()
            self.log("Camera released and window closed")
        except Exception as e:
            self.log(f"Camera cleanup error: {e}")
            
        self.camera_status_label.config(text="Camera: Stopped")

    def update_hw(self):
        try:
            # Update GPS coords
            lat, lon = gps_reader.get_coordinates()
            self.gps_lbl.config(text=f"{lat:.6f}, {lon:.6f}")

            # MPU
            x, y, z, mag = read_mpu6050()
            self.mpu_lbl.config(text=f"MPU6050: Mag={mag}")
            check_fall(mag)
            if mag < FALL_THRESHOLD_LOW or mag > FALL_THRESHOLD_HIGH:
                self.fall_lbl.config(text="Fall: ALERT POSSIBLE")
            else:
                self.fall_lbl.config(text="Fall: Safe")

            # Emergency
            if read_emergency():
                self.em_lbl.config(text="Emergency: ALERT Pressed")
                self.log("Emergency Button Pressed!")
                lat, lon = gps_reader.get_coordinates()
                msg = f"EMERGENCY: EMERGENCY BUTTON PRESSED! Location: {get_map_link(lat, lon)}"
                send_telegram(msg)
                send_email("EMERGENCY BUTTON PRESSED!", msg)
            else:
                self.em_lbl.config(text="Emergency: Safe")

            # Ultrasonics
            uv = {}
            now = time.time()
            for n in ULTRASONICS.keys():
                d = read_ultrasonic(n)
                uv[n] = d
                if d < ULTRA_THRESHOLD and now - last_ultra_alert_time[n] > ULTRA_COOLDOWN:
                    last_ultra_alert_time[n] = now
                    self.log(f"Ultrasonic Alert: {n} = {d}cm")
                    wav = SOUNDS_DIR / ULTRASONICS[n]["wav"]
                    play_sound_file(wav)
                    lat, lon = gps_reader.get_coordinates()
                    msg = f"ALERT: Obstacle {n}. Dist={d} at Loc: {get_map_link(lat, lon)}"
                    send_telegram(msg)
                    send_email(f"Obstacle Alert - {n}", msg)
            self.ultra_lbl.config(text=f"Ultrasonic: {uv}")

        except Exception as e:
            self.log(f"HW Error: {e}")

        # schedule next update
        self.root.after(500, self.update_hw)

# global alert dict
last_detection_alert = {}

# -----------------------------
# MAIN
# -----------------------------
def main():
    root = tk.Tk()
    app = GUI(root)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("Shutting down (KeyboardInterrupt)...")
    finally:
        try:
            gps_reader.stop()
        except Exception:
            pass
        try:
            GPIO.cleanup()
        except Exception:
            pass
        # Ensure all OpenCV windows are closed
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
