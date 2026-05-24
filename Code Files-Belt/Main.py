# pi_monitor.py - Enhanced GPS version v1.6 with fix awareness, satellite filtering, timeout, smoothing, GPRMC support, and debug logs

import os
import time
import sys
import threading
from queue import Queue
import webbrowser
import smtplib
import logging
import subprocess
from email.mime.text import MIMEText
from pathlib import Path
from collections import deque

# -----------------------------
# LOGGING SETUP
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("pi_monitor.log", encoding="utf-8"),
    ]
)
logger = logging.getLogger(__name__)

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
    logger.warning("PIL not available - camera preview disabled")

# picamera2 (Pi Camera Module) - preferred on modern Pi OS
try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
    logger.info("picamera2 available - Pi Camera Module will be used")
except Exception:
    PICAMERA2_AVAILABLE = False
    logger.warning("picamera2 not available - will try OpenCV (cv2.VideoCapture) as fallback")

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
    logger.warning("RPi.GPIO not available. Running in non-hardware (test) mode.")

try:
    import smbus
    SMBUS_AVAILABLE = True
except Exception:
    SMBUS_AVAILABLE = False
    logger.warning("smbus not available. MPU6050 functions will be disabled.")

# -----------------------------
# CONSTANTS / CONFIG
# -----------------------------
EMERGENCY_PIN = 17

ULTRASONICS = {
    "Left": {"TRIG": 23, "ECHO": 24, "wav": "left.wav"},
    "Right": {"TRIG": 27, "ECHO": 22, "wav": "right.wav"},
    "Back": {"TRIG": 6, "ECHO": 5, "wav": "back.wav"}
}

ULTRA_THRESHOLD = 30        # cm
ULTRA_COOLDOWN = 30          # seconds (increased from 5 to reduce spam)
last_ultra_alert_time = {k: 0 for k in ULTRASONICS.keys()}

last_ultra_sound_time = {k: 0 for k in ULTRASONICS.keys()}
SOUND_COOLDOWN = 2

ULTRA_SCAN_INTERVAL = 0.2   # 200 ms (recommended)
ultra_buffers = {k: deque(maxlen=3) for k in ULTRASONICS.keys()}
ultra_keys = list(ULTRASONICS.keys())
ultra_index = 0
last_ultra_scan_time = 0
last_ultra_values = {}  # Cache to retain ultrasonic values between scans

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "5610685031")
TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

# -----------------------------
# SMTP EMAIL SETTINGS (use environment variables or config)
# -----------------------------
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")     # e.g. smtp.gmail.com
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))                # e.g. 587 for STARTTLS
SMTP_USER = os.getenv("SMTP_USER", "ankit22csu216@ncuindia.edu") # your smtp username
SMTP_PASS = os.getenv("SMTP_PASS", "")    # your app password or SMTP password
ALERT_EMAIL = os.getenv("ALERT_EMAIL", "ankit22csu216@ncuindia.edu")  # recipient address

STATIC_LAT = 28.5039491
STATIC_LON = 77.0490655

SOUNDS_DIR = Path("sounds")
SOUNDS_DIR.mkdir(parents=True, exist_ok=True)

LAST_EMERGENCY_TIME = 0
EMERGENCY_COOLDOWN = 10

# GPS specific constants
GPS_TIMEOUT = 5  # seconds for stale data timeout

# ========================================
# ALERT QUEUE & WORKER (Thread-Safe)
# ========================================
alert_queue = Queue(maxsize=100)
last_alert_sent = {}  # Rate limiting per alert type
ALERT_MIN_INTERVAL = 10  # seconds between alerts of same type

def can_send(key):
    """Check if enough time has passed to send another alert of this type"""
    now = time.time()
    t = last_alert_sent.get(key, 0)
    if now - t >= ALERT_MIN_INTERVAL:
        last_alert_sent[key] = now
        return True
    return False

def enqueue_alert(kind, msg, subject="Alert"):
    """Non-blocking alert enqueueing (drops if queue full)"""
    try:
        alert_queue.put_nowait({
            "kind": kind,  # "telegram" | "email" | "both"
            "msg": msg,
            "subject": subject
        })
    except Exception:
        logger.warning("Alert queue full — dropping alert")

def alert_worker():
    """Background worker to send alerts from queue"""
    while True:
        item = alert_queue.get()
        if item is None:
            break  # graceful shutdown signal

        try:
            kind = item.get("kind")
            msg = item.get("msg")
            subject = item.get("subject", "Alert")

            if kind in ("telegram", "both"):
                send_telegram(msg)

            if kind in ("email", "both"):
                send_email(subject, msg)

        except Exception as e:
            logger.error(f"Alert worker error: {e}")
        finally:
            alert_queue.task_done()

# Start alert worker thread
alert_thread = threading.Thread(target=alert_worker, daemon=True)
alert_thread.start()
logger.info("Alert worker thread started")

# DNN model files (ensure they exist or skip model)
PROTOTXT = "models/MobileNetSSD_deploy.prototxt"
MODEL = "models/MobileNetSSD_deploy.caffemodel"

# -----------------------------
# Enhanced GPS Reader (threaded) - v1.7 with HDOP, safe reconnect, cold start
# -----------------------------
import serial

class GPSReader:
    def __init__(self):
        self.ser = None
        self.lat_in_degrees = STATIC_LAT
        self.lon_in_degrees = STATIC_LON
        self.gps_available = False
        self.has_fix = False
        self.running = False
        self.gps_thread = None
        self.last_gps_data_time = 0
        self.gps_data_count = 0
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        self.lat_buffer = deque(maxlen=5)
        self.lon_buffer = deque(maxlen=5)
        logger.info("Waiting for first GPS fix...")
        self.initialize_gps()

    def initialize_gps(self):
        """Initialize GPS with proper error handling"""
        logger.info("GPS: Cold start - waiting for first fix (may take 30s-5min)...")
        # Common GPS configurations for Raspberry Pi
        gps_configs = [
            {"port": "/dev/ttyAMA0", "baudrate": 9600},
            {"port": "/dev/serial0", "baudrate": 9600},
            {"port": "/dev/ttyS0", "baudrate": 9600},
            {"port": "/dev/ttyUSB0", "baudrate": 9600},
        ]
        
        for config in gps_configs:
            try:
                logger.info(f"GPS: Trying port {config['port']}...")
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
                
                if self.ser.is_open:
                    self.ser.reset_input_buffer()
                    self.ser.reset_output_buffer()
                    
                    test_line = self.ser.readline().decode('ascii', errors='ignore')
                    if test_line:
                        logger.info(f"GPS initialized: {config['port']}")
                        logger.debug(f"Test: {test_line.strip()}")
                        self.gps_available = True
                        self.has_fix = False  # Wait for valid data
                        break
                    else:
                        logger.warning(f"GPS {config['port']} no data")
                        self.ser.close()
                else:
                    logger.warning(f"Cannot open {config['port']}")
                    
            except Exception as e:
                logger.error(f"GPS init {config['port']}: {e}")
                if hasattr(self, 'ser') and self.ser and self.ser.is_open:
                    self.ser.close()
                self.ser = None
                
        if not self.gps_available:
            logger.warning("GPS: All ports failed. Using static location fallback.")
            logger.warning("Check wiring, raspi-config serial, and GPS power.")
            return
            
        self.running = True
        self.reconnect_attempts = 0
        if hasattr(self, 'gps_thread') and self.gps_thread and self.gps_thread.is_alive():
            logger.debug("Stopping old GPS thread before new...")
            self.running = False
            self.gps_thread.join(timeout=1.0)
        self.gps_thread = threading.Thread(target=self._gps_loop, daemon=True)
        self.gps_thread.start()
        logger.info("GPS v1.7 thread active")

    def _convert_to_degrees(self, raw_value):
        """Convert NMEA coordinates (DDmm.mmmm) to decimal degrees"""
        try:
            if not raw_value or raw_value == '':
                return 0.0
            raw_float = float(raw_value)
            degrees = int(raw_float / 100)
            minutes = raw_float - (degrees * 100)
            position = degrees + (minutes / 60.0)
            return round(position, 6)
        except Exception as e:
            logger.error(f"GPS degree conversion error for '{raw_value}': {e}")
            return 0.0

    def _parse_gps_data(self, received_data):
        """Parse NMEA GPS data - supports GPGGA and GPRMC with enhanced filtering"""
        try:
            # GPGGA parsing (primary)
            if "$GPGGA" in received_data:
                if self.gps_data_count < 10:  # Only log first 10 messages
                    logger.debug(f"GPS Raw GPGGA: {received_data[:80]}...")
                
                parts = received_data.split("$GPGGA,", 1)[1].split(',')
                
                if len(parts) >= 10:
                    nmea_lat = parts[1]
                    nmea_lat_dir = parts[2]
                    nmea_lon = parts[3]  
                    nmea_lon_dir = parts[4]
                    fix_quality = parts[5]
                    satellites = parts[6]
                    
                    if self.gps_data_count % 20 == 0:
                        logger.info(f"GPS Status - Fix: {fix_quality}, Sats: {satellites}")
                    
                    # Enhanced validation: fix + sats + HDOP
                    if (len(parts) >= 11 and fix_quality and fix_quality.isdigit() and int(fix_quality) > 0 and 
                        satellites and satellites.isdigit() and int(satellites) >= 4):
                        
                        hdop_str = parts[8] if len(parts) > 8 and parts[8] else "999"
                        try:
                            hdop = float(hdop_str)
                        except:
                            hdop = 999.0
                            
                        if hdop < 2.5:
                            if nmea_lat and nmea_lon and len(nmea_lat) > 0 and len(nmea_lon) > 0:
                                lat = self._convert_to_degrees(nmea_lat)
                                lon = self._convert_to_degrees(nmea_lon)
                            
                            if nmea_lat_dir == 'S':
                                lat = -lat
                            if nmea_lon_dir == 'W':
                                lon = -lon
                                
                            if -90 <= lat <= 90 and -180 <= lon <= 180:
                                logger.info(f"GPS VALID FIX: Lat={lat:.6f}, Lon={lon:.6f}, Sat={satellites}, Quality={fix_quality}")
                                self.has_fix = True
                                return lat, lon
                            else:
                                logger.warning(f"GPS: Invalid coordinates: lat={lat}, lon={lon}")
                        else:
                            logger.warning("GPS: Empty latitude/longitude data")
                    else:
                        self.has_fix = False
                        if self.gps_data_count % 10 == 0:
                            logger.debug(f"GPS: No valid fix (quality: {fix_quality}, sats: {satellites})")
                            
            # GPRMC parsing (backup/more reliable sometimes)
            elif "$GPRMC" in received_data:
                if self.gps_data_count < 10:
                    logger.debug(f"GPS Raw GPRMC: {received_data[:80]}...")
                
                parts = received_data.split("$GPRMC,", 1)[1].split(',')
                
                if len(parts) >= 9 and parts[2] == 'A':  # 'A' = active (valid fix)
                    nmea_lat = parts[3]
                    nmea_lat_dir = parts[4]
                    nmea_lon = parts[5]
                    nmea_lon_dir = parts[6]
                    
                    if nmea_lat and nmea_lon:
                        lat = self._convert_to_degrees(nmea_lat)
                        lon = self._convert_to_degrees(nmea_lon)
                        
                        if nmea_lat_dir == 'S':
                            lat = -lat
                        if nmea_lon_dir == 'W':
                            lon = -lon
                            
                        if -90 <= lat <= 90 and -180 <= lon <= 180:
                            logger.info(f"GPS GPRMC FIX: Lat={lat:.6f}, Lon={lon:.6f}")
                            self.has_fix = True
                            return lat, lon
                            
        except Exception as e:
            logger.error(f"GPS parsing error: {e}")
            
        self.has_fix = False
        return None, None

    def _gps_loop(self):
        """Main GPS reading loop with proper error recovery"""
        consecutive_errors = 0
        max_consecutive_errors = 5
        
        while self.running:
            try:
                if not self.ser or not self.ser.is_open:
                    if self.reconnect_attempts < self.max_reconnect_attempts:
                        logger.warning("GPS port closed, attempting to reconnect...")
                        self.reconnect_attempts += 1
                        self.initialize_gps()
                        time.sleep(2)
                        continue
                    else:
                        logger.error("Max GPS reconnect attempts reached. Giving up.")
                        break
                
                line = self.ser.readline().decode('ascii', errors='ignore').strip()
                if line:
                    consecutive_errors = 0
                    self.gps_data_count += 1
                    lat, lon = self._parse_gps_data(line)
                    if lat is not None and lon is not None:
                            # Apply moving average smoothing
                            self.lat_buffer.append(lat)
                            self.lon_buffer.append(lon)
                            self.lat_in_degrees = sum(self.lat_buffer) / len(self.lat_buffer)
                            self.lon_in_degrees = sum(self.lon_buffer) / len(self.lon_buffer)
                            self.last_gps_data_time = time.time()
                            if self.gps_data_count <= 10 or self.gps_data_count % 50 == 0:
                                logger.info(f"GPS SMOOTHED #{self.gps_data_count}: {self.lat_in_degrees:.6f}, {self.lon_in_degrees:.6f} (fix={self.has_fix})")
                else:
                    time.sleep(0.1)
                    
            except serial.SerialException as e:
                consecutive_errors += 1
                logger.error(f"GPS Serial error #{consecutive_errors}: {e}")
                if consecutive_errors >= max_consecutive_errors:
                    logger.error("Too many serial errors, attempting to reset GPS connection...")
                    try:
                        if self.ser and self.ser.is_open:
                            self.ser.close()
                    except Exception as close_err:
                        logger.warning(f"Error closing GPS serial port: {close_err}")
                    self.ser = None
                    consecutive_errors = 0
                    time.sleep(2)
                    
            except Exception as e:
                consecutive_errors += 1
                logger.error(f"GPS Unexpected error #{consecutive_errors}: {e}")
                if consecutive_errors >= max_consecutive_errors:
                    logger.error("Too many unexpected errors in GPS loop")
                    break
                time.sleep(1)

        logger.info("GPS thread stopped")

    def get_coordinates(self):
        """Get current coordinates with fix/timeout validation"""
        # Check timeout first
        if time.time() - self.last_gps_data_time > GPS_TIMEOUT:
            logger.debug("GPS data timeout - no valid fix")
            self.has_fix = False
            return None, None
        
        # Return valid fix coordinates or None
        if self.has_fix:
            return self.lat_in_degrees, self.lon_in_degrees
        else:
            logger.debug("No GPS fix available")
            return None, None

    def stop(self):
        """Stop GPS thread and close serial port"""
        self.running = False
        try:
            if self.ser and self.ser.is_open:
                self.ser.close()
                logger.info("GPS serial port closed")
        except Exception as e:
            logger.error(f"GPS stop error: {e}")

gps_reader = GPSReader()

# -----------------------------
# Telegram helper
# -----------------------------
def send_telegram(msg):
    try:
        data = {"chat_id": TELEGRAM_CHAT_ID, "text": msg}
        requests.post(TELEGRAM_API_URL, data=data, timeout=3)
        logger.info(f"Telegram sent: {msg}")
    except Exception as e:
        logger.error(f"Telegram send failed: {e}")

# -----------------------------
# EMAIL HELPER (SMTP)
# -----------------------------
def send_email(subject, body):
    try:
        msg = MIMEText(body, "plain")
        msg["Subject"] = subject
        msg["From"] = SMTP_USER
        msg["To"] = ALERT_EMAIL

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=10) as server:
            server.ehlo()
            try:
                server.starttls()
                server.ehlo()
            except Exception as tls_err:
                logger.warning(f"STARTTLS not supported, proceeding without it: {tls_err}")
            server.login(SMTP_USER, SMTP_PASS)
            server.sendmail(SMTP_USER, [ALERT_EMAIL], msg.as_string())

        logger.info(f"EMAIL SENT: {subject} -> {ALERT_EMAIL}")
    except Exception as e:
        logger.error(f"Email send failed: {e}")

# -----------------------------
# GPIO setup (safe)
# -----------------------------
if ON_PI:
    GPIO.setwarnings(False)
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
        logger.info("MPU6050 initialized")
    except Exception as e:
        logger.warning(f"MPU6050 not available: {e}")
        mpu_available = False

def _mpu_read_word(reg: int) -> float:
    """Read a signed 16-bit word from MPU6050 register pair."""
    h = bus.read_byte_data(MPU_ADDR, reg)
    l = bus.read_byte_data(MPU_ADDR, reg + 1)
    v = (h << 8) + l
    if v >= 0x8000:
        return float(-((65535 - v) + 1))
    return float(v)

def read_mpu6050():
    if not mpu_available:
        return 0.0, 0.0, 0.0, 1.0
    try:
        x = _mpu_read_word(0x3B) / 16384.0
        y = _mpu_read_word(0x3D) / 16384.0
        z = _mpu_read_word(0x3F) / 16384.0
        mag = round((x*x + y*y + z*z) ** 0.5, 3)
        return round(x, 2), round(y, 2), round(z, 2), mag
    except Exception as e:
        logger.error(f"MPU read error: {e}")
        return 0.0, 0.0, 0.0, 1.0

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
        if lat is None or lon is None:
            lat, lon = STATIC_LAT, STATIC_LON
    return f"https://www.google.com/maps?q={lat},{lon}"

def check_fall(mag):
    global last_fall_alert
    if not mpu_available or mag == 0.0:
        return
    now = time.time()
    if (mag < FALL_THRESHOLD_LOW or mag > FALL_THRESHOLD_HIGH) and (now - last_fall_alert > FALL_COOLDOWN):
        lat, lon = gps_reader.get_coordinates()
        if lat is None or lon is None:
            lat, lon = STATIC_LAT, STATIC_LON
        msg = f"ALERT: Fall Detected! Mag={mag}\\nLocation: {get_map_link(lat, lon)}"
        enqueue_alert("both", msg, "Fall Detected!")
        last_fall_alert = now
        logger.warning(f"Fall detected! Magnitude: {mag}")

# -----------------------------
# Ultrasonic (with timeouts)
# -----------------------------
def read_ultrasonic(name, timeout=0.025):  # balanced speed/accuracy
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
        while GPIO.input(s["ECHO"]) == 0:
            if time.time() - start_time > timeout:
                return 999.0
        start = time.time()

        stop = None
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
        logger.error(f"Ultrasonic read error: {e}")
        return 999.0

def read_emergency():
    if not ON_PI:
        return False
    try:
        return GPIO.input(EMERGENCY_PIN) == GPIO.LOW
    except Exception:
        return False

# -----------------------------
# Object detection DNN (optional)
# -----------------------------
CLASSES = ["background","aeroplane","bicycle","bird","boat","bottle","bus",
           "car","cat","chair","cow","diningtable","dog","horse","motorbike",
           "person","pottedplant","sheep","sofa","train","tvmonitor"]

DANGEROUS = {"car","dog","train"}
net = None
if os.path.exists(PROTOTXT) and os.path.exists(MODEL):
    try:
        net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
        logger.info("DNN loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load DNN: {e}")
        net = None
else:
    logger.warning("DNN files not found; continuing without object detection.")
    logger.warning(f"Looking for: {PROTOTXT} and {MODEL}")

def play_sound_file(path: Path):
    try:
        if path.exists():
            subprocess.Popen(["aplay", "-q", str(path)],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        logger.error(f"Play sound error: {e}")

# -----------------------------
# GUI Class
# -----------------------------
class GUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("SATRA - Smart Alert & Tracking for Risk Avoidance")
        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()

        root.geometry(f"{screen_w}x{screen_h}")
        root.minsize(1024, 600)
        try:
            root.state("zoomed")
        except:
            root.attributes("-zoomed", True)
        root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Configure theme
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(".", background="#1e1e1e", foreground="white")
        style.configure("TLabelframe", background="#1e1e1e", foreground="white")
        style.configure("TLabelframe.Label", background="#1e1e1e", foreground="#00ffcc")
        style.configure("TLabel", background="#1e1e1e", foreground="white")
        style.configure("TButton", padding=6)
        style.configure("Header.TLabel", font=("Arial", 15, "bold"))
        style.configure("Title.TLabel", font=("Arial", 10, "bold"))
        style.configure("Info.TLabel", font=("Arial", 11))
        style.configure("Card.TFrame", relief="solid", borderwidth=1, background="#2a2a2a")
        style.configure("CardSafe.TFrame", background="#1b4d1b")
        style.configure("CardWarn.TFrame", background="#6f5500")
        style.configure("CardAlert.TFrame", background="#6d1e1e")
        style.configure("CardOffline.TFrame", background="#2a2a2a")
        style.configure("Footer.TFrame", background="#171717")
        style.configure("Footer.TLabel", background="#171717", foreground="#9effff")
        style.configure("CameraStatus.TLabel", background="#1e1e1e", foreground="#ff0000", font=("Arial", 11, "bold"))

        # Main container with grid layout
        container = ttk.Frame(root, padding=8)
        container.pack(fill="both", expand=True)
        container.columnconfigure(0, weight=1, uniform="group1")
        container.columnconfigure(1, weight=3, uniform="group1")
        container.rowconfigure(1, weight=1)
        container.rowconfigure(2, weight=0)
        container.rowconfigure(3, weight=0)

        # ===== HEADER =====
        header = ttk.Frame(container)
        header.grid(row=0, column=0, columnspan=2, sticky="ew", pady=5)

        ttk.Label(header, text="SATRA - Smart Alert & Tracking for Risk Avoidance", style="Header.TLabel").pack(side="left", padx=5)
        self.time_lbl = ttk.Label(header, font=("Arial", 11))
        self.time_lbl.pack(side="left", padx=20)
        self.update_clock()

        # Quick action buttons
        btn_frame = ttk.Frame(header)
        btn_frame.pack(side="right", padx=5)
        self.start_cam_btn = ttk.Button(btn_frame, text="[START] Start Cam", command=self.start_camera)
        self.start_cam_btn.pack(side="left", padx=3)
        self.stop_cam_btn = ttk.Button(btn_frame, text="[STOP] Stop Cam", command=self.stop_camera)
        self.stop_cam_btn.pack(side="left", padx=3)
        self.stop_cam_btn.state(["disabled"])
        ttk.Button(btn_frame, text="[MAP] Map", command=lambda: webbrowser.open(get_map_link())).pack(side="left", padx=3)
        ttk.Button(btn_frame, text="[GPS] GPS Status", command=self.show_gps_status).pack(side="left", padx=3)

        # ===== STATUS PANEL (LEFT) =====
        status_frame = ttk.LabelFrame(container, text="System Status", padding=6)
        status_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        status_frame.columnconfigure(0, weight=1)
        status_frame.columnconfigure(1, weight=1)

        # Helper to create status cards
        def create_status_card(parent, title, row, col):
            frame = ttk.Frame(parent, style="Card.TFrame", relief="solid", padding=8, borderwidth=1)
            frame.grid(row=row, column=col, sticky="ew", padx=4, pady=4)
            label_title = tk.Label(frame, text=title, font=("Arial", 10, "bold"), bg="#2a2a2a", fg="white")
            label_title.pack(anchor="w")
            value = tk.Label(frame, text="--", font=("Arial", 11), bg="#2a2a2a", fg="white")
            value.pack(anchor="w")
            value.card_frame = frame
            return frame, value

        self.mpu_card, self.mpu_lbl = create_status_card(status_frame, "Accelerometer (MPU6050)", 0, 0)
        self.fall_card, self.fall_lbl = create_status_card(status_frame, "Fall Detection", 0, 1)
        self.ultra_card, self.ultra_lbl = create_status_card(status_frame, "Ultrasonic Sensors", 1, 0)
        self.em_card, self.em_lbl = create_status_card(status_frame, "Emergency Button", 1, 1)
        self.gps_status_card, self.gps_status_lbl = create_status_card(status_frame, "GPS Status", 2, 0)
        self.gps_coords_card, self.gps_lbl = create_status_card(status_frame, "GPS Coordinates", 2, 1)

        self.gps_status_lbl.config(text="Active" if gps_reader.gps_available else "Not Available")
        self.gps_lbl.config(text="Coordinates: --,--")

        # ===== CAMERA PANEL (RIGHT) =====
        camera_frame = ttk.LabelFrame(container, text="Camera Feed", padding=10)
        camera_frame.grid(row=1, column=1, sticky="nsew", padx=5, pady=5)

        self.camera_status_dot = ttk.Label(camera_frame, text="[●] Offline", style="CameraStatus.TLabel")
        self.camera_status_dot.pack(anchor="w", pady=5)
        
        self.camera_canvas = tk.Canvas(
            camera_frame,
            bg="black",
            highlightthickness=0
        )
        self.camera_canvas.pack(fill="both", expand=True, pady=5)
        self.tk_image = None  # Prevent garbage collection of PhotoImage
        self.current_frame = None

        # ===== LOGS PANEL (BOTTOM) =====
        log_frame = ttk.LabelFrame(container, text="System Logs", padding=10)
        log_frame.grid(row=2, column=0, columnspan=2, sticky="nsew", pady=5)
        container.rowconfigure(2, weight=0)

        self.log_box = tk.Text(log_frame, height=7, state="disabled", bg="#1e1e1e", fg="#00ff00", font=("Courier", 9))
        self.log_box.pack(fill="both", expand=True, pady=5)

        # Scrollbar for logs
        scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_box.yview)
        scrollbar.pack(side="right", fill="y")
        self.log_box["yscrollcommand"] = scrollbar.set

        footer_frame = ttk.Frame(container, padding=6, style="Footer.TFrame")
        footer_frame.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(0, 5))
        footer_frame.columnconfigure(0, weight=1)
        self.footer_status_lbl = ttk.Label(footer_frame, text="System Online | Camera Active | GPS Fix | Alerts Enabled", style="Footer.TLabel")
        self.footer_status_lbl.pack(side="left", padx=5)

        self.camera_running = False
        self.cap = None
        self.external_window_open = False
        self.last_status_values = {}  # Track changes to avoid redundant updates

        self.root.after(100, self.update_hw)
        self.log("[INIT] System initialized")

    def on_closing(self):
        self.log("Shutting down...")
        self.camera_running = False
        try:
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()
        except Exception:
            pass
        gps_reader.stop()
        try:
            GPIO.cleanup()
        except Exception:
            pass
        # Graceful alert queue shutdown
        try:
            alert_queue.put(None)  # Signal worker to stop
        except Exception:
            pass
        self.root.destroy()

    def log(self, msg):
        """Log message with timestamp, auto-limit to 200 lines"""
        t = time.strftime("%H:%M:%S")
        self.log_box.config(state="normal")
        self.log_box.insert("end", f"[{t}] {msg}\n")

        # Limit to last 200 lines to prevent memory issues
        line_count = int(self.log_box.index('end-1c').split('.')[0])
        if line_count > 200:
            self.log_box.delete("1.0", "2.0")

        self.log_box.see("end")
        self.log_box.config(state="disabled")

    def update_gui_frame(self):
        try:
            if hasattr(self, "current_frame") and self.current_frame is not None:
                frame = self.current_frame

                canvas_w = self.camera_canvas.winfo_width()
                canvas_h = self.camera_canvas.winfo_height()

                h, w = frame.shape[:2]

                scale = min(canvas_w / w, canvas_h / h)

                new_w = int(w * scale)
                new_h = int(h * scale)

                frame = cv2.resize(frame, (new_w, new_h))

                img = Image.fromarray(frame)
                self.tk_image = ImageTk.PhotoImage(image=img)

                x = (canvas_w - new_w) // 2
                y = (canvas_h - new_h) // 2

                self.camera_canvas.delete("all")
                self.camera_canvas.create_image(
                    x,
                    y,
                    anchor="nw",
                    image=self.tk_image
                )

        except Exception as e:
            self.log(f"GUI frame update error: {e}")
        
    def update_clock(self):
        now = time.strftime("%H:%M:%S")
        self.time_lbl.config(text=now)
        self.root.after(1000, self.update_clock)

    def show_gps_status(self):
        lat, lon = gps_reader.get_coordinates()
        
        status = "HAS FIX" if gps_reader.has_fix else "NO FIX"
        
        if lat is not None and lon is not None:
            coords = f"{lat:.6f}, {lon:.6f}"
        else:
            coords = "No valid coordinates"

        msg = f"[GPS] Status: {status} | Coords: {coords} | Count: {gps_reader.gps_data_count}"
        
        self.log(msg)

    def set_status(self, label, text, status="ok"):
        """Update status label and card background with color coding"""
        colors = {
            "ok": "#00ff00",      # Green
            "warn": "#ffcc00",    # Orange
            "alert": "#ff6f6f",   # Red
            "offline": "#999999"  # Gray
        }
        card_bg = {
            "ok": "#1b4d1b",
            "warn": "#6f5500",
            "alert": "#6d1e1e",
            "offline": "#2a2a2a"
        }
        label.config(text=text, fg=colors.get(status, "white"))
        if hasattr(label, "card_frame"):
            bg = card_bg.get(status, "#2a2a2a")
            label.card_frame.config(style=f"Card{status.capitalize()}.TFrame")
            for child in label.card_frame.winfo_children():
                if isinstance(child, tk.Label):
                    child.config(bg=bg)

    def update_camera_status(self, running):
        """Update camera status indicator"""
        if running:
            self.camera_status_dot.config(text="[●] Online", foreground="#00ff00")
        else:
            self.camera_status_dot.config(text="[●] Offline", foreground="#ff0000")

    def start_camera(self):
        if self.camera_running:
            return
        self.camera_running = True
        self.update_camera_status(True)
        t = threading.Thread(target=self.cam_loop, daemon=True)
        t.start()
        self.start_cam_btn.state(["disabled"])
        self.stop_cam_btn.state(["!disabled"])
        if PICAMERA2_AVAILABLE:
            self.log("[CAM] Camera started using picamera2")
        else:
            self.log("[CAM] Camera started using OpenCV")

    def stop_camera(self):
        self.camera_running = False
        self.update_camera_status(False)
        self.log("[CAM] Camera stopped")
        self.start_cam_btn.state(["!disabled"])
        self.stop_cam_btn.state(["disabled"])
        self.external_window_open = False

    def _open_camera(self):
        """Initialize and open camera (picamera2 or OpenCV)"""
        if PICAMERA2_AVAILABLE:
            try:
                picam2 = Picamera2()
                config = picam2.create_preview_configuration(
                    main={"size": (640, 480), "format": "RGB888"},
                    controls={"FrameRate": 20}
                )
                picam2.configure(config)
                picam2.start()
                time.sleep(0.5)
                self.log("picamera2 started (640x480)")
                return picam2, None
            except Exception as e:
                self.log(f"picamera2 failed: {e}")

        for index in [0, 1, 2]:
            cap = cv2.VideoCapture(index)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
                cap.set(cv2.CAP_PROP_FPS, 30)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self.log(f"OpenCV camera {index} (320x240)")
                return None, cap
            cap.release()
        self.log("No camera available")
        return None, None

    def cam_loop(self):
        TARGET_FPS = 15
        FRAME_MS = 1.0 / TARGET_FPS
        DNN_EVERY_N = 60
        frame_count = 0
        last_detections = []

        picam2, cap = self._open_camera()
        if picam2 is None and cap is None:
            self.camera_running = False
            self.root.after(0, lambda: self.start_cam_btn.state(["!disabled"]))
            self.root.after(0, lambda: self.stop_cam_btn.state(["disabled"]))
            return

        try:
            while self.camera_running:
                t_start = time.time()
                try:
                    if picam2 is not None:
                        rgb = picam2.capture_array()
                        frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    else:
                        ret, frame = cap.read()
                        if not ret or frame is None:
                            time.sleep(0.1)
                            continue

                    if net is not None and frame_count % DNN_EVERY_N == 0:
                        try:
                            (h, w) = frame.shape[:2]
                            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (224, 224)), 0.007843, (224, 224), 127.5)
                            net.setInput(blob)
                            detections = net.forward()

                            last_detections = []
                            for i in range(detections.shape[2]):
                                confidence = detections[0, 0, i, 2]
                                if confidence > 0.4:
                                    idx = int(detections[0, 0, i, 1])
                                    if idx >= len(CLASSES):
                                        continue
                                    label = CLASSES[idx]
                                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                                    last_detections.append((label, box.astype("int"), confidence))

                                    wavpath = SOUNDS_DIR / f"{label}.wav"
                                    if wavpath.exists():
                                        subprocess.Popen(["aplay", "-q", str(wavpath)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                                    now = time.time()
                                    with _detection_alert_lock:
                                        last_alert_time = last_detection_alert.get(label, 0)
                                    if label.lower() in DANGEROUS and now - last_alert_time > 30:
                                        lat, lon = gps_reader.get_coordinates()
                                        if lat is None or lon is None:
                                            lat, lon = STATIC_LAT, STATIC_LON
                                        msg = f"ALERT: Dangerous {label} at {get_map_link(lat, lon)}"
                                        if can_send(f"danger_{label}"):
                                            enqueue_alert("both", msg, "Danger Alert")
                                        with _detection_alert_lock:
                                            last_detection_alert[label] = now

                        except Exception as e:
                            self.log(f"DNN error: {e}")

                    for (label, box, conf) in last_detections:
                        (startX, startY, endX, endY) = box
                        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                        cv2.putText(frame, f"{label} {conf:.2f}", (startX, max(startY - 10, 0)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

                    elapsed = time.time() - t_start
                    live_fps = 1.0 / elapsed if elapsed > 0 else 0
                    cv2.putText(frame, f"FPS:{live_fps:.1f} F:{frame_count}", (4, 16),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)

                    # Convert and display frame in canvas
                    if PIL_AVAILABLE:
                        try:
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                            # pass frame safely to main thread
                            self.current_frame = frame_rgb
                            self.root.after(0, self.update_gui_frame)

                        except Exception as e:
                            self.log(f"Canvas render error: {e}")
        
                    if frame_count % 60 == 0:
                        self.log(f"Camera: frame={frame_count}, fps={live_fps:.1f}, detections={len(last_detections)}")

                    frame_count += 1

                except Exception as e:
                    self.log(f"Camera loop error: {e}")
                    time.sleep(0.1)
                    continue

                spent = time.time() - t_start
                sleep_for = FRAME_MS - spent
                if sleep_for > 0:
                    time.sleep(sleep_for)

        finally:
            if picam2 is not None:
                try:
                    picam2.stop()
                    picam2.close()
                except Exception as e:
                    self.log(f"picamera2 close error: {e}")
            if cap is not None:
                try:
                    cap.release()
                except Exception as e:
                    self.log(f"OpenCV release error: {e}")

    def update_hw(self):
        try:
            lat, lon = gps_reader.get_coordinates()
            if lat is not None:
                self.set_status(self.gps_lbl, f"Coordinates: {lat:.6f}, {lon:.6f}", "ok")
                self.set_status(self.gps_status_lbl, "HAS FIX", "ok")
            else:
                self.set_status(self.gps_lbl, "Coordinates: Waiting for fix...", "warn")
                self.set_status(self.gps_status_lbl, "NO FIX", "alert")

            x, y, z, mag = read_mpu6050()
            if mpu_available:
                self.set_status(self.mpu_lbl, f"Mag={mag:.1f}m/s²", "ok")
                check_fall(mag)
                if mag < FALL_THRESHOLD_LOW or mag > FALL_THRESHOLD_HIGH:
                    self.set_status(self.fall_lbl, "ALERT - Fall Detected!", "alert")
                else:
                    self.set_status(self.fall_lbl, "Safe", "ok")
            else:
                self.set_status(self.mpu_lbl, "Offline", "offline")
                self.set_status(self.fall_lbl, "Offline", "offline")

            if read_emergency():
                global LAST_EMERGENCY_TIME
                now = time.time()
                self.set_status(self.em_lbl, "ALERT!", "alert")
                if now - LAST_EMERGENCY_TIME > EMERGENCY_COOLDOWN:
                    LAST_EMERGENCY_TIME = now
                    self.log("[ALERT] EMERGENCY BUTTON PRESSED!")
                    lat, lon = gps_reader.get_coordinates()
                    if lat is None or lon is None:
                        lat, lon = STATIC_LAT, STATIC_LON
                    msg = f"EMERGENCY! Location: {get_map_link(lat, lon)}"
                    enqueue_alert("both", msg, "EMERGENCY!")
            else:
                self.set_status(self.em_lbl, "Safe", "ok")


            global last_ultra_scan_time, last_ultra_values, ultra_index, last_ultra_sound_time
            uv = last_ultra_values.copy()
            now = time.time()

            # Scan only one sensor per loop (round-robin)
            if now - last_ultra_scan_time >= ULTRA_SCAN_INTERVAL:
                last_ultra_scan_time = now

                n = ultra_keys[ultra_index]
                ultra_index = (ultra_index + 1) % len(ultra_keys)
                d = read_ultrasonic(n)

                # Filtering: add to buffer if not error, else clear buffer
                if d != 999.0:
                    ultra_buffers[n].append(d)
                else:
                    ultra_buffers[n].clear()

                # Use average if buffer has data
                if len(ultra_buffers[n]) > 0:
                    d = sum(ultra_buffers[n]) / len(ultra_buffers[n])
                else:
                    d = 999.0

                if 2 <= d <= ULTRA_THRESHOLD:
                    uv[n] = d
                    last_ultra_values[n] = d

                    # Play sound for all directions (with cooldown)
                    if now - last_ultra_sound_time[n] > SOUND_COOLDOWN:
                        play_sound_file(SOUNDS_DIR / ULTRASONICS[n]["wav"])
                        last_ultra_sound_time[n] = now

                    # ALERT ONLY FOR BACK
                    if n == "Back":
                        if now - last_ultra_alert_time[n] > ULTRA_COOLDOWN:
                            last_ultra_alert_time[n] = now

                            self.log(f"[WARN] Obstacle BACK: {d}cm")

                            lat, lon = gps_reader.get_coordinates()
                            if lat is None:
                                lat, lon = STATIC_LAT, STATIC_LON

                            msg = f"BACK Obstacle: {d}cm at {get_map_link(lat, lon)}"
                            enqueue_alert("both", msg, "Back Obstacle Alert")
                else:
                    if n in last_ultra_values:
                        del last_ultra_values[n]

            # Update ultrasonic status with color coding
            ultra_status = "ok" if len(uv) == 0 else "warn" if len(uv) <= 1 else "alert"
            self.set_status(self.ultra_lbl, f"{str(uv) if uv else 'All clear'}", ultra_status if uv else "ok")

        except Exception as e:
            self.log(f"HW Error: {e}")

        self.root.after(100, self.update_hw)

# Global alert protection
last_detection_alert = {}
_detection_alert_lock = threading.Lock()

# -----------------------------
# MAIN
# -----------------------------
def main():
    root = tk.Tk()
    app = GUI(root)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        logger.info("Shutdown (Ctrl+C)")
    finally:
        gps_reader.stop()
        try:
            GPIO.cleanup()
        except Exception:
            pass
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

