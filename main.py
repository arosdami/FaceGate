import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import time
import pickle
import os
import sys
from pathlib import Path
import hashlib
import json
import base64
import platform
import subprocess
from datetime import datetime
import logging

# Platform-specifikus importok
try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False

class CrossPlatformFaceGate:
    def __init__(self):
        # Statikus ablaknevek - NEM változnak nyelvváltáskor
        self.WINDOW_SECURITY = "FaceGate Security"
        self.WINDOW_REGISTRATION = "Face Registration"
        
        # Nyelvi beállítások
        self.languages = {
            'hungarian': self.setup_hungarian_texts(),
            'english': self.setup_english_texts()
        }
        self.current_language = 'hungarian'
        
        self.text = self.languages[self.current_language]
        
        # Rendszer információk
        self.system_info = {
            'developer': 'Damjan Aros - THE PTI',
            'project': 'Egyetemi Projekt - FaceGate',
            'version': '2.0',
            'course': 'Robotika, MI & NN',
            'university': 'Tokaj-Hegyalja Egyetem',
            'supervisor': 'Attila Perlaki, David Gegeny',
            'semester': '2025 Os'
        }
        
        self.setup_logging()
        self.detect_platform()
        self.setup_directories()
        
        # Komponensek inicializálása
        self.face_cascade = self.load_cascade('haarcascade_frontalface_default.xml')
        self.eye_cascade = self.load_cascade('haarcascade_eye.xml')
        self.mouth_cascade = self.load_cascade('haarcascade_smile.xml')
        
        self.cnn_model = None
        self.lbph_recognizer = None
        self.initialize_recognizer()
        
        self.face_db = {}
        self.lbph_trained = False
        
        # Beállítások alapértelmezett értékekkel
        self.settings = {
            'threshold': 0.98,
            'unlock_duration': 10,
            'camera_index': 0,
            'arduino_port': self.get_default_serial_port(),
            'max_samples': 30,
            'min_confidence': 0.85,
            'enable_anti_spoofing': False,
            'enable_sound': True,
            'language': 'hungarian'
        }
        
        self.arduino = None
        self.available_cameras = []
        self.available_serial_ports = []
        
        self.encryption_key = self.generate_encryption_key()
        
    def setup_hungarian_texts(self):
        return {
            # Főmenü
            'main_menu_title': "PROFESSIONAL FACEGATE BIZTONSAGI RENDSZER",
            'menu_options': {
                '1': "Uj arc regisztralasa (Titkosított)",
                '2': "Biztonsagi rendszer inditasa",
                '3': "Rendszer beallitasok",
                '4': "Rendszer informaciok",
                '5': "Adatbazis kezeles",
                '6': "Nyelv valtas",
                '7': "Kilepes"
            },
            'select_option': "Valassz opciot (1-7): ",
            'invalid_choice': "[HIBA] Ervenytelen valasztas",
            
            # Rendszer információk
            'system_info_title': "RENDSZER INFORMACIOK",
            'developer': "Fejleszto",
            'project': "Projekt",
            'version': "Verzio",
            'course': "Kurzus",
            'university': "Egyetem",
            'supervisor': "Temavezeto",
            'semester': "Felev",
            'platform': "Platform",
            'cameras_detected': "Kamerak eszlelve",
            'serial_ports': "Soros portok",
            
            # Kamera választás
            'camera_selection': "KAMERA KIVALASZTAS",
            'available_cameras': "Elerheto kamerak",
            'camera': "Kamera",  
            'select_camera': "Valassz kamerat",
            'camera_selected': "Kamera kivalasztva",
            'progress': "Haladas",  
            'status': "Allapot", 
            'identity': "Azonosit",  
            'method': "Modszer",  
            'last': "Utolso", 
            'door_unlocked': "Ajto kinyitva",  
            'door_locked': "Ajto bezarva",  
            'no_database': "Nincs adatbazis",  
            'persons': "szemely",  
            'images': "kep",  
            'database': "adatbazis",  
            'confidence': "Biztonsagi szint",  
            'system_shutdown': "RENDSZER LEALLITAS", 
            
            # Arduino beállítás
            'arduino_setup': "ARDUINO BEALLITAS",
            'available_ports': "Elerheto portok",
            'enter_port': "Add meg az Arduino portot (Enter az alapértelmezettre): ",
            'arduino_connected': "Arduino csatlakoztatva",
            'arduino_not_found': "Arduino nem talalhato",
            'simulation_mode': "Szimulacios modban fut",
            
            # Arc regisztráció
            'face_registration': "ARC REGISZTRACIO",
            'enter_name': "Add meg a szemely nevet: ",
            'invalid_name': "[HIBA] Ervenytelen nev",
            'starting_registration': "Titkositott regisztracio inditasa",
            'landmark_detection': "ARC JELLEMZOPONTOK ESZLELESE",
            'detecting_features': "Szemek, szaj es arc jellemzok eszlelese...",
            'generating_points': "Veletlenszeru biztonsagi pontok generalasa...",
            'encryption': "Titkositas",
            'generating_embeddings': "Titkositott beagyazasok generalasa...",
            'registration_success': "Sikeresen regisztralva",
            'samples_collected': "Mintak gyujtve",
            'landmark_signatures': "Jellemzopont alairasok",
            'data_stored': "Minden adat XOR titkosítassal tarolva",
            'not_enough_samples': "Nincs eleg jo minosegu minta",
            
            # Biztonsági rendszer
            'security_system': "BIZTONSAGI RENDSZER",
            'system_activated': "Biztonsagi rendszer aktiválva",
            'exit_instructions': "Nyomj 'q'-t a kilepeshez, 'l'-t a kezi zarashoz",
            'scanning': "SCANNING",
            'encrypted_recognition': "Titkositott arcfelismeres inditva...",
            'access_granted': "HOZZAFERES ENGEDELYEZVE",
            'encrypted_match': "Titkositott jellemzopont egyezes",
            'scanning_timeout': "SCANNING TIMEOUT - Nem talalhato titkositott egyezes",
            'auto_lock': "AUTOMATA ZARAS - Ajto bezarva",
            'manual_lock': "KEZI ZARAS - Ajto bezarva",
            
            # Általános
            'success': "SIKERES",
            'error': "HIBA",
            'warning': "FIGYELMEZTETES",
            'info': "INFORMACIO",
            'loading': "Betoltes",
            'saving': "Mentes",
            'encrypted': "Titkositott",
            'unknown': "ISMERETLEN",
            'no_face': "NINCS ARC",
            'scanning_text': "SCANNING...",
            'locked': "ZARVA",
            'unlocked': "NYITVA",
            'exit': "Kilepes",
            'back': "Vissza",
            
            # Beállítások menü
            'settings_title': "RENDSZER BEALLITASOK",
            'current_settings': "Jelenlegi beallitasok",
            'change_settings': "Beallitas modositasa",
            'setting_options': {
                '1': "Kuszobertek modositasa",
                '2': "Nyitva tartasi ido modositasa",
                '3': "Kamera modositasa",
                '4': "Arduino port modositasa",
                '5': "Mintaszam modositasa",
                '6': "Minimalis biztonsagi szint",
                '7': "Anti-spoofing bekapcsolasa",
                '8': "Hang effektek",
                '9': "Vissza a főmenube"
            },
            'threshold_prompt': "Uj kuszobertek (jelenleg: {}): ",
            'duration_prompt': "Uj nyitva tartasi ido (masodperc, jelenleg: {}): ",
            'samples_prompt': "Uj mintaszam (jelenleg: {}): ",
            'confidence_prompt': "Uj minimalis biztonsagi szint (jelenleg: {}): ",
            'anti_spoofing_prompt': "Anti-spoofing engedelyezese (i/n, jelenleg: {}): ",
            'sound_prompt': "Hang effektek engedelyezese (i/n, jelenleg: {}): ",
            'settings_saved': "Beallitasok elmentve",
            
            # Beállítások kulcsok
            'threshold': "Kuszobertek",
            'unlock_duration': "Nyitva tartasi ido", 
            'arduino_port': "Arduino port",
            'max_samples': "Mintaszam",
            'min_confidence': "Minimalis biztonsagi szint",
            'enable_anti_spoofing': "Anti-spoofing",
            'enable_sound': "Hang effektek",
            
            # Adatbázis kezelés
            'database_title': "ADATBAZIS KEZELES",
            'database_options': {
                '1': "Regisztralt szemelyek listazasa",
                '2': "Szemely torlese",
                '3': "Adatbazis biztonsagi mentes",
                '4': "Adatbazis visszaallitas",
                '5': "Statisztikak megjelenitese",
                '6': "Vissza a főmenube"
            },
            'registered_persons': "Regisztralt szemelyek",
            'no_persons': "Nincsenek regisztralt szemelyek",
            'delete_prompt': "Add meg a torolni kivant szemely nevet: ",
            'person_not_found': "Szemely nem talalhato",
            'person_deleted': "Szemely torolve",
            'backup_created': "Biztonsagi mentes letrehozva",
            'restore_complete': "Visszaallitas befejezve",
            'database_stats': "Adatbazis statisztikak",
            'total_persons': "Osszes szemely",
            'total_samples': "Osszes minta",
            'db_size': "Adatbazis meret",
            
            # Nyelv választás
            'language_title': "NYELV VALASZTAS",
            'current_language': "Jelenlegi nyelv",
            'select_language': "Valassz nyelvet",
            'language_options': {
                '1': "Magyar",
                '2': "English"
            },
            'language_changed': "Nyelv megvaltoztatva",
            
            # Rendszer státusz
            'system_status': "RENDSZER ALLAPOT",
            'camera_status': "Kamera",
            'arduino_status': "Arduino",
            'registered_count': "Regisztralt szemelyek",
            'encryption_status': "Titkositas",
            'recognition_method': "Felismeresi modszer",
            'security_threshold': "Biztonsagi kuszob",
            'all_data_encrypted': "Minden arcadat titkositott jellemzopontokkal tarolva"
        }
    
    def setup_english_texts(self):
        return {
            # Main Menu
            'main_menu_title': "PROFESSIONAL FACEGATE SECURITY SYSTEM",
            'menu_options': {
                '1': "Register New Face (Encrypted)",
                '2': "Start Security System",
                '3': "System Settings",
                '4': "System Information",
                '5': "Database Management",
                '6': "Change Language",
                '7': "Exit"
            },
            'select_option': "Select option (1-7): ",
            'invalid_choice': "[ERROR] Invalid selection",
            
            # System Information
            'system_info_title': "SYSTEM INFORMATION",
            'developer': "Developer",
            'project': "Project",
            'version': "Version",
            'course': "Course",
            'university': "University",
            'supervisor': "Supervisor",
            'semester': "Semester",
            'platform': "Platform",
            'cameras_detected': "Cameras detected",
            'serial_ports': "Serial ports",
            
            # Camera Selection
            'camera_selection': "CAMERA SELECTION",
            'available_cameras': "Available cameras",
            'camera': "Camera",
            'select_camera': "Select camera",
            'camera_selected': "Camera selected",
            'progress': "Progress",
            'status': "Status",
            'identity': "Identity",
            'method': "Method",
            'last': "Last",
            'door_unlocked': "Door unlocked",
            'door_locked': "Door locked",
            'no_database': "No database",
            'persons': "persons",
            'images': "images",
            'database': "database",
            'confidence': "Confidence",
            'system_shutdown': "SYSTEM SHUTDOWN",
            
            # Arduino Setup
            'arduino_setup': "ARDUINO SETUP",
            'available_ports': "Available ports",
            'enter_port': "Enter Arduino port (Enter for default): ",
            'arduino_connected': "Arduino connected",
            'arduino_not_found': "Arduino not found",
            'simulation_mode': "Running in simulation mode",
            
            # Face Registration
            'face_registration': "FACE REGISTRATION",
            'enter_name': "Enter person name: ",
            'invalid_name': "[ERROR] Invalid name",
            'starting_registration': "Starting encrypted registration",
            'landmark_detection': "FACIAL LANDMARK DETECTION",
            'detecting_features': "Detecting eyes, mouth and facial features...",
            'generating_points': "Generating randomized security points...",
            'encryption': "Encryption",
            'generating_embeddings': "Generating encrypted embeddings...",
            'registration_success': "Successfully registered",
            'samples_collected': "Samples collected",
            'landmark_signatures': "Landmark signatures",
            'data_stored': "All data stored with XOR encryption",
            'not_enough_samples': "Not enough high-quality samples",
            
            # Security System
            'security_system': "SECURITY SYSTEM",
            'system_activated': "Security system activated",
            'exit_instructions': "Press 'q' to exit, 'l' to manually lock",
            'scanning': "SCANNING",
            'encrypted_recognition': "Encrypted face recognition started...",
            'access_granted': "ACCESS GRANTED",
            'encrypted_match': "Encrypted landmark match",
            'scanning_timeout': "SCANNING TIMEOUT - No encrypted match found",
            'auto_lock': "AUTO LOCK - Door locked",
            'manual_lock': "MANUAL LOCK - Door locked",
            
            # General
            'success': "SUCCESS",
            'error': "ERROR",
            'warning': "WARNING",
            'info': "INFO",
            'loading': "Loading",
            'saving': "Saving",
            'encrypted': "Encrypted",
            'unknown': "UNKNOWN",
            'no_face': "NO FACE",
            'scanning_text': "SCANNING...",
            'locked': "LOCKED",
            'unlocked': "UNLOCKED",
            'exit': "Exit",
            'back': "Back",
            
            # Settings Menu
            'settings_title': "SYSTEM SETTINGS",
            'current_settings': "Current settings", 
            'change_settings': "Change settings",
            'setting_options': {
                '1': "Change threshold value",
                '2': "Change unlock duration",
                '3': "Change camera",
                '4': "Change Arduino port", 
                '5': "Change sample count",
                '6': "Change minimum confidence",
                '7': "Enable anti-spoofing",
                '8': "Sound effects",
                '9': "Back to main menu"
            },
            'threshold_prompt': "New threshold (current: {}): ",
            'duration_prompt': "New unlock duration (seconds, current: {}): ",
            'samples_prompt': "New sample count (current: {}): ",
            'confidence_prompt': "New minimum confidence (current: {}): ",
            'anti_spoofing_prompt': "Enable anti-spoofing (y/n, current: {}): ",
            'sound_prompt': "Enable sound effects (y/n, current: {}): ",
            'settings_saved': "Settings saved",
            
            # Settings keys
            'threshold': "Threshold",
            'unlock_duration': "Unlock duration",
            'arduino_port': "Arduino port",
            'max_samples': "Sample count", 
            'min_confidence': "Minimum confidence",
            'enable_anti_spoofing': "Anti-spoofing",
            'enable_sound': "Sound effects",
            
            # Database Management
            'database_title': "DATABASE MANAGEMENT",
            'database_options': {
                '1': "List registered persons",
                '2': "Delete person",
                '3': "Backup database",
                '4': "Restore database",
                '5': "Show statistics",
                '6': "Back to main menu"
            },
            'registered_persons': "Registered persons",
            'no_persons': "No registered persons",
            'delete_prompt': "Enter name to delete: ",
            'person_not_found': "Person not found",
            'person_deleted': "Person deleted",
            'backup_created': "Backup created",
            'restore_complete': "Restore completed",
            'database_stats': "Database statistics",
            'total_persons': "Total persons",
            'total_samples': "Total samples",
            'db_size': "Database size",
            
            # Language Selection
            'language_title': "LANGUAGE SELECTION",
            'current_language': "Current language",
            'select_language': "Select language",
            'language_options': {
                '1': "Hungarian",
                '2': "English"
            },
            'language_changed': "Language changed",
            
            # System Status
            'system_status': "SYSTEM STATUS",
            'camera_status': "Camera",
            'arduino_status': "Arduino",
            'registered_count': "Registered persons",
            'encryption_status': "Encryption",
            'recognition_method': "Recognition method",
            'security_threshold': "Security threshold",
            'all_data_encrypted': "All facial data stored with encrypted landmarks"
        }
    
    def change_language(self, language):
        if language in self.languages:
            self.current_language = language
            self.settings['language'] = language
            self.text = self.languages[language]
            self.logger.info(f"Language changed to {language}")
            return True
        return False
    
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('facegate_system.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('FaceGate')
        
    def detect_platform(self):
        self.platform = platform.system().lower()
        self.logger.info(f"Detected platform: {self.platform}")
        
    def setup_directories(self):
        directories = ['data', 'models', 'logs', 'exports', 'backups']
        for dir_name in directories:
            Path(dir_name).mkdir(exist_ok=True)
            
    def load_cascade(self, cascade_name):
        try:
            cascade_path = cv2.data.haarcascades + cascade_name
            cascade = cv2.CascadeClassifier(cascade_path)
            if cascade.empty():
                self.logger.warning(f"Could not load {cascade_name}, trying alternative path")
                alternative_paths = [
                    f"/usr/share/opencv4/haarcascades/{cascade_name}",
                    f"/usr/local/share/opencv4/haarcascades/{cascade_name}",
                    f"./haarcascades/{cascade_name}"
                ]
                for path in alternative_paths:
                    cascade = cv2.CascadeClassifier(path)
                    if not cascade.empty():
                        self.logger.info(f"Loaded {cascade_name} from {path}")
                        return cascade
            else:
                self.logger.info(f"Loaded {cascade_name} successfully")
                return cascade
        except Exception as e:
            self.logger.error(f"Error loading {cascade_name}: {e}")
            
        self.logger.warning(f"Using basic cascade for {cascade_name}")
        return cv2.CascadeClassifier()
    
    def initialize_recognizer(self):
        try:
            self.lbph_recognizer = cv2.face.LBPHFaceRecognizer.create()
        except AttributeError:
            try:
                self.lbph_recognizer = cv2.face.LBPHFaceRecognizer_create()
            except AttributeError:
                self.logger.warning("LBPH recognizer not available")
                self.lbph_recognizer = None
    
    def get_default_serial_port(self):
        if self.platform == 'windows':
            return 'COM4'
        elif self.platform == 'darwin':  # macOS
            return '/dev/cu.usbmodem14101'
        else:  # Linux
            return '/dev/ttyUSB0'
    
    def detect_available_cameras(self):
        self.available_cameras = []
        max_test_cameras = 10
        
        self.logger.info("Scanning for available cameras...")
        for i in range(max_test_cameras):
            try:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        self.available_cameras.append(i)
                        self.logger.info(f"Found camera {i}: {frame.shape[1]}x{frame.shape[0]}")
                    cap.release()
                else:
                    cap.release()
            except Exception as e:
                self.logger.debug(f"Camera {i} not available: {e}")
                
        if not self.available_cameras:
            self.logger.warning("No cameras detected!")
        else:
            self.logger.info(f"Found {len(self.available_cameras)} cameras")
    
    def detect_serial_ports(self):
        self.available_serial_ports = []
        
        if not SERIAL_AVAILABLE:
            self.logger.warning("PySerial not available, serial port detection skipped")
            return
            
        if self.platform == 'windows':
            ports = [f'COM{i}' for i in range(1, 257)]
        elif self.platform == 'darwin':
            ports = [f'/dev/cu.usbmodem{i:03d}' for i in range(1, 257)] + \
                   [f'/dev/ttyUSB{i}' for i in range(10)]
        else:  # Linux
            ports = [f'/dev/ttyUSB{i}' for i in range(10)] + \
                   [f'/dev/ttyACM{i}' for i in range(10)]
        
        for port in ports:
            try:
                ser = serial.Serial(port)
                ser.close()
                self.available_serial_ports.append(port)
                self.logger.info(f"Found serial port: {port}")
            except (serial.SerialException, OSError):
                pass
                
        if not self.available_serial_ports:
            self.logger.info("No serial ports detected")

    def generate_encryption_key(self):
        key_file = Path("data/encryption_key.key")
        if not key_file.exists():
            key = os.urandom(32)
            with open(key_file, 'wb') as f:
                f.write(key)
            self.logger.info("New encryption key generated")
        else:
            with open(key_file, 'rb') as f:
                key = f.read()
            self.logger.info("Existing encryption key loaded")
        return key

    def detect_facial_landmarks(self, face_roi):
        gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        landmarks = {
            'eyes': [],
            'mouth': [],
            'face_shape': [],
            'random_points': []
        }
        
        eyes = self.eye_cascade.detectMultiScale(gray_face, 1.1, 5)
        for (ex, ey, ew, eh) in eyes:
            center_x = ex + ew // 2
            center_y = ey + eh // 2
            landmarks['eyes'].append((center_x, center_y))
            
            for _ in range(3):
                rx = center_x + np.random.randint(-10, 10)
                ry = center_y + np.random.randint(-5, 5)
                landmarks['random_points'].append((rx, ry))
        
        mouths = self.mouth_cascade.detectMultiScale(gray_face, 1.5, 15)
        for (mx, my, mw, mh) in mouths:
            center_x = mx + mw // 2
            center_y = my + mh // 2
            landmarks['mouth'].append((center_x, center_y))
            
            for _ in range(3):
                rx = center_x + np.random.randint(-15, 15)
                ry = center_y + np.random.randint(-5, 5)
                landmarks['random_points'].append((rx, ry))
        
        h, w = gray_face.shape
        face_points = [
            (10, 10), (w-10, 10), (w//2, h-10),
            (w//4, h//3), (3*w//4, h//3),
            (w//4, 2*h//3), (3*w//4, 2*h//3)
        ]
        landmarks['face_shape'] = face_points
        
        for _ in range(10):
            rx = np.random.randint(5, w-5)
            ry = np.random.randint(5, h-5)
            landmarks['random_points'].append((rx, ry))
        
        return landmarks
    
    def create_face_signature(self, face_roi, landmarks):
        all_points = []
        for category in ['eyes', 'mouth', 'face_shape', 'random_points']:
            all_points.extend(landmarks[category])
        
        points_array = np.array(all_points, dtype=np.float32)
        
        if len(points_array) > 0:
            points_array = points_array / np.array([face_roi.shape[1], face_roi.shape[0]])
        
        noise = np.random.normal(0, 0.01, points_array.shape)
        points_array += noise
        
        signature = points_array.flatten()
        if len(signature) < 128:
            signature = np.pad(signature, (0, 128 - len(signature)))
        
        return signature[:128]
    
    def build_cnn_model(self):
        model = keras.Sequential([
            layers.Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
            layers.MaxPooling2D((2,2)),
            layers.Conv2D(64, (3,3), activation='relu'),
            layers.MaxPooling2D((2,2)),
            layers.Conv2D(128, (3,3), activation='relu'),
            layers.MaxPooling2D((2,2)),
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(128, activation='sigmoid')
        ])
        
        model.compile(optimizer='adam', loss='cosine_similarity')
        return model

    def initialize_system(self):
        self.show_header()
        self.detect_available_cameras()
        self.detect_serial_ports()
        self.select_camera()
        self.setup_arduino()
        
        if not self.cnn_model:
            self.cnn_model = self.build_cnn_model()
            
        self.load_database()
        self.load_lbph_model()
        print(f"\n[{self.text['success']}] Professional FaceGate System Initialized!")
        
    def show_header(self):
        print("=" * 70)
        print(f"           {self.text['main_menu_title']}")
        print("=" * 70)
        print(f"Developer: {self.system_info['developer']}")
        print(f"Platform: {self.platform.upper()} | Language: {self.current_language.upper()}")
        print("=" * 70)
        
    def select_camera(self):
        print(f"\n[{self.text['camera_selection']}]")
        
        if not self.available_cameras:
            print(f"[{self.text['warning']}] No cameras detected!")
            return
            
        print(f"{self.text['available_cameras']}:")
        for i, cam_idx in enumerate(self.available_cameras):
            print(f"  {cam_idx} - {self.text['camera']} {cam_idx}")
        
        while True:
            try:
                choice = input(f"{self.text['select_camera']} ({', '.join(map(str, self.available_cameras))}): ").strip()
                if choice == '' and self.available_cameras:
                    self.settings['camera_index'] = self.available_cameras[0]
                    break
                elif choice.isdigit() and int(choice) in self.available_cameras:
                    self.settings['camera_index'] = int(choice)
                    break
                else:
                    print(f"[{self.text['error']}] {self.text['invalid_choice']}")
            except KeyboardInterrupt:
                return
                
        print(f"[{self.text['success']}] {self.text['camera_selected']}: {self.settings['camera_index']}")
                
    def setup_arduino(self):
        print(f"\n[{self.text['arduino_setup']}]")
        
        if not self.available_serial_ports:
            print(f"[{self.text['info']}] {self.text['arduino_not_found']}")
            print(f"[{self.text['info']}] {self.text['simulation_mode']}")
            self.arduino = None
            return
            
        print(f"{self.text['available_ports']}: {', '.join(self.available_serial_ports)}")
        
        try:
            port = input(f"{self.text['enter_port']}").strip()
            if port:
                self.settings['arduino_port'] = port
            else:
                self.settings['arduino_port'] = self.get_default_serial_port()
                
            self.arduino = serial.Serial(self.settings['arduino_port'], 9600, timeout=1)
            time.sleep(2)
            
            while self.arduino.in_waiting > 0:
                self.arduino.readline()
                
            print(f"[{self.text['success']}] {self.text['arduino_connected']} {self.settings['arduino_port']}")
            
        except Exception as e:
            print(f"[{self.text['warning']}] {self.text['arduino_not_found']}: {e}")
            print(f"[{self.text['info']}] {self.text['simulation_mode']}")
            self.arduino = None

    def load_database(self):
        try:
            with open('data/secure_facegate_database.pkl', 'rb') as f:
                encrypted_db = pickle.load(f)
            
            self.face_db = {}
            for name, encrypted_data in encrypted_db.items():
                decrypted_data = self.simple_decrypt(encrypted_data)
                if decrypted_data is not None:
                    self.face_db[name] = decrypted_data
            
            print(f"[{self.text['success']}] {self.text['encrypted']} {self.text['database']} {self.text['loading']}: {len(self.face_db)} {self.text['persons']}")
        except Exception as e:
            print(f"[{self.text['info']}] {self.text['no_database']}: {e}")
            self.face_db = {}
            
    def save_database(self):
        encrypted_db = {}
        for name, data in self.face_db.items():
            encrypted_db[name] = self.simple_encrypt(data)
        
        with open('data/secure_facegate_database.pkl', 'wb') as f:
            pickle.dump(encrypted_db, f)
        
        self.train_lbph_model()
        print(f"[{self.text['success']}] {self.text['all_data_encrypted']}")
        
    def train_lbph_model(self):
        if not self.face_db or self.lbph_recognizer is None:
            return
            
        faces = []
        labels = []
        current_label = 0
        
        for name in self.face_db.keys():
            face_dir = Path(f"./data/secure_faces/{name}")
            if face_dir.exists():
                for img_path in face_dir.glob("*.jpg"):
                    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        img = cv2.resize(img, (100, 100))
                        faces.append(img)
                        labels.append(current_label)
            current_label += 1
        
        if faces and len(set(labels)) > 0:
            self.lbph_recognizer.train(faces, np.array(labels))
            self.lbph_recognizer.write('models/secure_facegate_lbph_model.xml')
            self.lbph_trained = True
            print(f"[{self.text['success']}] LBPH model trained with {len(faces)} {self.text['images']}")

    def register_face(self):
        print(f"\n[{self.text['face_registration']}]")
        name = input(f"{self.text['enter_name']}").strip()
        if not name:
            print(f"[{self.text['error']}] {self.text['invalid_name']}")
            return
            
        print(f"\n{self.text['starting_registration']}: {name}")
        print(f"{self.text['landmark_detection']}")
        
        face_dir = Path(f"./data/secure_faces/{name}")
        face_dir.mkdir(parents=True, exist_ok=True)
        
        # Ablak bezárása előző sessionből
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        
        cap = cv2.VideoCapture(self.settings['camera_index'])
        cnn_samples = []
        landmark_signatures = []
        total_samples = self.settings['max_samples']
        
        print(f"\n[{self.text['landmark_detection']}]")
        print(f"{self.text['detecting_features']}")
        print(f"{self.text['generating_points']}")
        
        collected = 0
        try:
            while collected < total_samples:
                ret, frame = cap.read()
                if not ret:
                    continue
                    
                face, bbox, analysis, landmarks = self.detect_and_analyze_face(frame)
                display_frame = frame.copy()
                
                if face is not None and analysis['quality'] == "HIGH":
                    signature = self.create_face_signature(face, landmarks)
                    landmark_signatures.append(signature)
                    cnn_samples.append(face)
                    
                    x, y, w, h = bbox
                    gray_face = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
                    gray_face = cv2.resize(gray_face, (100, 100))
                    cv2.imwrite(str(face_dir / f"{collected}.jpg"), gray_face)
                    
                    collected += 1
                    
                    self.draw_landmarks(display_frame, bbox, landmarks)
                    cv2.rectangle(display_frame, (x,y), (x+w,y+h), (0,255,0), 3)
                    
                    # Statikus szövegek - nem változnak nyelvváltáskor
                    status_text = "ENCRYPTING LANDMARKS..."
                    cv2.putText(display_frame, status_text, (20, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                    cv2.putText(display_frame, f"Progress: {collected}/{total_samples}", (20, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
                
                self.draw_registration_ui(display_frame, name, collected, total_samples)
                # STATIKUS ablaknév használata
                cv2.imshow(self.WINDOW_REGISTRATION, display_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
        finally:
            cap.release()
            cv2.destroyWindow(self.WINDOW_REGISTRATION)
            cv2.waitKey(1)
        
        if len(cnn_samples) >= 20:
            print(f"\n[{self.text['encryption']}] {self.text['generating_embeddings']}")
            
            cnn_embeddings = []
            for sample in cnn_samples:
                embedding = self.cnn_model.predict(np.array([sample]), verbose=0)[0]
                cnn_embeddings.append(embedding)
            
            combined_data = {
                'cnn_embeddings': cnn_embeddings,
                'landmark_signatures': landmark_signatures,
                'registration_time': time.time(),
                'sample_count': len(cnn_samples)
            }
            
            self.face_db[name] = combined_data
            self.save_database()
            
            print(f"[{self.text['success']}] {name} {self.text['registration_success']}!")
            print(f"{self.text['samples_collected']}: {len(cnn_samples)}")
            print(f"{self.text['landmark_signatures']}: {len(landmark_signatures)}")
            print(f"{self.text['data_stored']}")
        else:
            print(f"[{self.text['error']}] {self.text['not_enough_samples']}")
            
    def detect_and_analyze_face(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=6, 
            minSize=(120,120)
        )
        
        if len(faces) == 0:
            return None, None, {}, {}
            
        x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
        face_roi = frame[y:y+h, x:x+w]
        
        landmarks = self.detect_facial_landmarks(face_roi)
        
        gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray_face)
        contrast = np.std(gray_face)
        sharpness = cv2.Laplacian(gray_face, cv2.CV_64F).var()
        
        eyes = self.eye_cascade.detectMultiScale(gray_face, 1.1, 3)
        
        if brightness > 80 and contrast > 40 and sharpness > 50 and len(eyes) >= 1:
            quality = "HIGH"
        else:
            quality = "LOW"
        
        analysis = {
            'quality': quality,
            'brightness': brightness,
            'contrast': contrast,
            'eyes_detected': len(eyes)
        }
        
        face_processed = cv2.resize(face_roi, (128,128))
        face_processed = cv2.cvtColor(face_processed, cv2.COLOR_BGR2RGB)
        face_processed = face_processed.astype('float32') / 255.0
        
        return face_processed, (x,y,w,h), analysis, landmarks
    
    def draw_landmarks(self, frame, bbox, landmarks):
        x, y, w, h = bbox
        
        for px, py in landmarks['eyes']:
            abs_x, abs_y = x + px, y + py
            cv2.circle(frame, (abs_x, abs_y), 3, (0, 255, 0), -1)
        
        for px, py in landmarks['mouth']:
            abs_x, abs_y = x + px, y + py
            cv2.circle(frame, (abs_x, abs_y), 3, (255, 0, 0), -1)
        
        for px, py in landmarks['face_shape']:
            abs_x, abs_y = x + px, y + py
            cv2.circle(frame, (abs_x, abs_y), 2, (0, 0, 255), -1)
        
        for px, py in landmarks['random_points']:
            abs_x, abs_y = x + px, y + py
            cv2.circle(frame, (abs_x, abs_y), 1, (0, 255, 255), -1)
    
    def draw_registration_ui(self, frame, name, current, total):
        h, w = frame.shape[:2]
        
        cv2.rectangle(frame, (0,0), (w,100), (0,0,0), -1)
        
        # Statikus szövegek - nem változnak nyelvváltáskor
        title = f"REGISTRATION: {name}"
        status = "ENCRYPTING FACIAL LANDMARKS..."
            
        cv2.putText(frame, title, (20,30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(frame, status, (20,60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
        cv2.putText(frame, f"Progress: {current}/{total}", (20,85), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        
        progress = int((current/total) * (w-40))
        cv2.rectangle(frame, (20, h-40), (w-20, h-20), (50,50,50), -1)
        cv2.rectangle(frame, (20, h-40), (20+progress, h-20), (0,255,0), -1)
        
        # Statikus footer
        footer = "XOR Encrypted Landmark Storage"
            
        cv2.putText(frame, footer, 
                   (w-400, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 1)

    def security_system(self):
        print(f"\n[{self.text['security_system']}]")
        print(f"{self.text['system_activated']}")
        print(f"{self.text['exit_instructions']}")
        
        # Ablak bezárása előző sessionből
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        
        cap = cv2.VideoCapture(self.settings['camera_index'])
        door_unlocked = False
        unlock_time = 0
        last_person = self.text['unknown']
        unknown_counter = 0
        is_scanning = False
        scanning_start_time = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    continue
                    
                if door_unlocked and time.time() - unlock_time > self.settings['unlock_duration']:
                    self.lock_door()
                    door_unlocked = False
                    print(f"[{self.text['auto_lock']}]")
                    unknown_counter = 0
                    is_scanning = False
                    
                face, bbox, analysis, landmarks = self.detect_and_analyze_face(frame)
                
                if face is not None and not door_unlocked and not is_scanning:
                    is_scanning = True
                    scanning_start_time = time.time()
                    print(f"[{self.text['scanning']}] {self.text['encrypted_recognition']}")
                
                if is_scanning and face is not None:
                    name, confidence, method = self.secure_face_recognition(face, landmarks, frame)
                    
                    if (name != self.text['unknown'] and name != self.text['scanning_text'] and 
                        confidence > self.settings['threshold'] and not door_unlocked):
                        
                        print(f"[{self.text['access_granted']}] {name} - {self.text['confidence']}: {confidence:.3f}")
                        print(f"{self.text['encrypted_match']}: {method}")
                        if self.unlock_door():
                            door_unlocked = True
                            unlock_time = time.time()
                            last_person = name
                            unknown_counter = 0
                            is_scanning = False
                    
                    elif time.time() - scanning_start_time > 5:
                        print(f"[{self.text['scanning_timeout']}]")
                        is_scanning = False
                        unknown_counter += 1
                
                if is_scanning:
                    display_name = self.text['scanning_text']
                    display_confidence = 0.0
                    display_method = "ENCRYPTED_MATCH"
                else:
                    if face is not None:
                        name, confidence, method = self.secure_face_recognition(face, landmarks, frame)
                        display_name = name
                        display_confidence = confidence
                        display_method = method
                    else:
                        display_name = self.text['no_face']
                        display_confidence = 0.0
                        display_method = "NO_FACE"
                
                self.draw_security_ui(frame, display_name, display_confidence, 
                                   bbox, door_unlocked, last_person, 
                                   display_method, unlock_time, is_scanning, landmarks)
                # STATIKUS ablaknév használata
                cv2.imshow(self.WINDOW_SECURITY, frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('l') and door_unlocked:
                    self.lock_door()
                    door_unlocked = False
                    print(f"[{self.text['manual_lock']}]")
        finally:
            if door_unlocked:
                self.lock_door()
            cap.release()
            cv2.destroyWindow(self.WINDOW_SECURITY)
            cv2.waitKey(1)
    
    def secure_face_recognition(self, face_processed, landmarks, frame):
        if not self.face_db:
            return "NO_DATA", 0.0, "NO_DATA"
        
        current_signature = self.create_face_signature(face_processed, landmarks)
        query_embedding = self.cnn_model.predict(np.array([face_processed]), verbose=0)[0]
        
        best_match = self.text['unknown']
        best_similarity = 0.0
        best_method = "NO_MATCH"
        
        for name, stored_data in self.face_db.items():
            cnn_similarity = 0.0
            for emb in stored_data['cnn_embeddings']:
                similarity = self.cosine_similarity(query_embedding, emb)
                if similarity > cnn_similarity:
                    cnn_similarity = similarity
            
            landmark_similarity = 0.0
            for stored_sig in stored_data['landmark_signatures']:
                similarity = self.cosine_similarity(current_signature, stored_sig)
                if similarity > landmark_similarity:
                    landmark_similarity = similarity
            
            combined_similarity = (cnn_similarity + landmark_similarity) / 2
            
            if combined_similarity > best_similarity:
                best_similarity = combined_similarity
                best_match = name
                best_method = "ENCRYPTED_LANDMARKS" if landmark_similarity > cnn_similarity else "CNN_EMBEDDING"
        
        return best_match, best_similarity, best_method
    
    def cosine_similarity(self, a, b):
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
        if a_norm == 0 or b_norm == 0:
            return 0.0
        return np.dot(a, b) / (a_norm * b_norm)
        
    def unlock_door(self):
        if self.arduino:
            try:
                self.arduino.write(b"UNLOCK\n")
                time.sleep(0.5)
                print(f"[ARDUINO] {self.text['door_unlocked']}")
                return True
            except Exception as e:
                print(f"[{self.text['error']}] Arduino communication: {e}")
                return False
        else:
            print(f"[SIMULATION] {self.text['door_unlocked']}")
            return True
            
    def lock_door(self):
        if self.arduino:
            try:
                self.arduino.write(b"LOCK\n")
                time.sleep(0.5)
                print(f"[ARDUINO] {self.text['door_locked']}")
                return True
            except Exception as e:
                print(f"[{self.text['error']}] Arduino communication: {e}")
                return False
        else:
            print(f"[SIMULATION] {self.text['door_locked']}")
            return True
            
    def draw_security_ui(self, frame, name, confidence, bbox, unlocked, last_person, method, unlock_time, is_scanning, landmarks=None):
        h, w = frame.shape[:2]
        
        if is_scanning:
            bg_color = (0, 50, 100)
        elif name == self.text['unknown']:
            bg_color = (0, 0, 100)
        else:
            bg_color = (0, 0, 0)
            
        cv2.rectangle(frame, (0,0), (w,140), bg_color, -1)
        
        status_color = (0,255,0) if unlocked else (0,0,255)
        if unlocked:
            status_text = "UNLOCKED"
        elif is_scanning:
            status_text = "SCANNING..."
        else:
            status_text = "LOCKED"
            
        cv2.putText(frame, f"STATUS: {status_text}", (20,30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        
        if is_scanning:
            name_color = (0,255,255)
            name_text = "ENCRYPTED SCANNING..."
        else:
            name_color = (0,255,0) if name not in [self.text['unknown'], self.text['no_face'], "NO_DATA"] else (0,0,255)
            name_text = name
            
        cv2.putText(frame, f"IDENTITY: {name_text}", (20,65), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, name_color, 2)
        
        if not is_scanning and name not in [self.text['no_face'], self.text['scanning_text']]:
            cv2.putText(frame, f"CONFIDENCE: {confidence:.3f}", (20,90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            cv2.putText(frame, f"METHOD: {method}", (20,110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        
        if bbox and landmarks:
            self.draw_landmarks(frame, bbox, landmarks)
            x, y, w_rect, h_rect = bbox
            box_color = (0,255,255) if is_scanning else (0,255,0) if name not in [self.text['unknown'], self.text['no_face']] else (0,0,255)
            cv2.rectangle(frame, (x,y), (x+w_rect,y+h_rect), box_color, 3)
            
            if is_scanning:
                for i in range(8):
                    offset = (int(time.time() * 10) + i) % 30
                    cv2.circle(frame, (x + offset, y + 10), 5, (0,255,255), -1)
        
        cv2.rectangle(frame, (0,h-40), (w,h), (0,0,0), -1)
        cv2.putText(frame, f"Last: {last_person}", (20, h-15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        
        # Statikus vezérlők
        controls = "Q: Exit | L: Lock"
        encryption_status = "Encrypted Landmarks Active"
            
        cv2.putText(frame, controls, (w-150, h-15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        cv2.putText(frame, encryption_status, (w-300, h-15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 1)
        
        if unlocked:
            elapsed = time.time() - unlock_time
            remaining = max(0, self.settings['unlock_duration'] - elapsed)
            
            auto_lock_text = f"AUTO LOCK: {remaining:.1f}s"
                
            cv2.putText(frame, auto_lock_text, (w-200, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

    def show_system_information(self):
        print(f"\n[{self.text['system_info_title']}]")
        print("=" * 50)
        print(f"{self.text['developer']}: {self.system_info['developer']}")
        print(f"{self.text['project']}: {self.system_info['project']}")
        print(f"{self.text['version']}: {self.system_info['version']}")
        print(f"{self.text['course']}: {self.system_info['course']}")
        print(f"{self.text['university']}: {self.system_info['university']}")
        print(f"{self.text['supervisor']}: {self.system_info['supervisor']}")
        print(f"{self.text['semester']}: {self.system_info['semester']}")
        print(f"{self.text['platform']}: {self.platform.upper()}")
        print(f"{self.text['cameras_detected']}: {len(self.available_cameras)}")
        print(f"{self.text['serial_ports']}: {len(self.available_serial_ports)}")
        print(f"{self.text['current_language']}: {self.current_language.upper()}")
        print("=" * 50)

    def show_system_status(self):
        print(f"\n[{self.text['system_status']}]")
        print(f"{self.text['camera_status']}: {self.settings['camera_index']}")
        print(f"{self.text['arduino_status']}: {'Connected' if self.arduino else 'Disconnected'}")
        print(f"{self.text['registered_count']}: {len(self.face_db)}")
        print(f"{self.text['encryption_status']}: XOR Active")
        print(f"{self.text['recognition_method']}: CNN + {self.text['encrypted']} Landmarks")
        print(f"{self.text['security_threshold']}: {self.settings['threshold']}")
        print(f"{self.text['all_data_encrypted']}")

    def settings_menu(self):
        while True:
            print(f"\n[{self.text['settings_title']}]")
            print("=" * 50)
            print(f"{self.text['current_settings']}:")
            print(f"  1. {self.text['threshold']}: {self.settings['threshold']}")
            print(f"  2. {self.text['unlock_duration']}: {self.settings['unlock_duration']}s")
            print(f"  3. {self.text['camera']}: {self.settings['camera_index']}")
            print(f"  4. {self.text['arduino_port']}: {self.settings['arduino_port']}")
            print(f"  5. {self.text['max_samples']}: {self.settings['max_samples']}")
            print(f"  6. {self.text['min_confidence']}: {self.settings['min_confidence']}")
            print(f"  7. {self.text['enable_anti_spoofing']}: {self.settings['enable_anti_spoofing']}")
            print(f"  8. {self.text['enable_sound']}: {self.settings['enable_sound']}")
            print(f"  9. {self.text['back']}")
            print("=" * 50)
            
            choice = input(f"{self.text['select_option']}").strip()
            
            if choice == '1':
                new_threshold = input(self.text['threshold_prompt'].format(self.settings['threshold']))
                try:
                    self.settings['threshold'] = float(new_threshold)
                    print(f"[{self.text['success']}] {self.text['threshold']} updated")
                except ValueError:
                    print(f"[{self.text['error']}] Invalid value")
                    
            elif choice == '2':
                new_duration = input(self.text['duration_prompt'].format(self.settings['unlock_duration']))
                try:
                    self.settings['unlock_duration'] = int(new_duration)
                    print(f"[{self.text['success']}] {self.text['unlock_duration']} updated")
                except ValueError:
                    print(f"[{self.text['error']}] Invalid value")
                    
            elif choice == '3':
                self.select_camera()
                
            elif choice == '4':
                self.setup_arduino()
                
            elif choice == '5':
                new_samples = input(self.text['samples_prompt'].format(self.settings['max_samples']))
                try:
                    self.settings['max_samples'] = int(new_samples)
                    print(f"[{self.text['success']}] {self.text['max_samples']} updated")
                except ValueError:
                    print(f"[{self.text['error']}] Invalid value")
                    
            elif choice == '6':
                new_confidence = input(self.text['confidence_prompt'].format(self.settings['min_confidence']))
                try:
                    self.settings['min_confidence'] = float(new_confidence)
                    print(f"[{self.text['success']}] {self.text['min_confidence']} updated")
                except ValueError:
                    print(f"[{self.text['error']}] Invalid value")
                    
            elif choice == '7':
                current = "Igen" if self.settings['enable_anti_spoofing'] else "Nem"
                if self.current_language == 'english':
                    current = "Yes" if self.settings['enable_anti_spoofing'] else "No"
                response = input(self.text['anti_spoofing_prompt'].format(current)).lower()
                self.settings['enable_anti_spoofing'] = response in ['i', 'y', 'igen', 'yes']
                print(f"[{self.text['success']}] Anti-spoofing updated")
                
            elif choice == '8':
                current = "Igen" if self.settings['enable_sound'] else "Nem"
                if self.current_language == 'english':
                    current = "Yes" if self.settings['enable_sound'] else "No"
                response = input(self.text['sound_prompt'].format(current)).lower()
                self.settings['enable_sound'] = response in ['i', 'y', 'igen', 'yes']
                print(f"[{self.text['success']}] Sound settings updated")
                
            elif choice == '9':
                break
            else:
                print(f"[{self.text['error']}] {self.text['invalid_choice']}")

    def database_management(self):
        while True:
            print(f"\n[{self.text['database_title']}]")
            print("=" * 50)
            for key, value in self.text['database_options'].items():
                print(f"  {key}. {value}")
            print("=" * 50)
            
            choice = input(f"{self.text['select_option']}").strip()
            
            if choice == '1':
                self.list_registered_persons()
            elif choice == '2':
                self.delete_person()
            elif choice == '3':
                self.backup_database()
            elif choice == '4':
                self.restore_database()
            elif choice == '5':
                self.show_database_stats()
            elif choice == '6':
                break
            else:
                print(f"[{self.text['error']}] {self.text['invalid_choice']}")

    def list_registered_persons(self):
        print(f"\n[{self.text['registered_persons']}]")
        if not self.face_db:
            print(f"  {self.text['no_persons']}")
        else:
            for i, name in enumerate(self.face_db.keys(), 1):
                samples = self.face_db[name]['sample_count']
                print(f"  {i}. {name} ({samples} samples)")

    def delete_person(self):
        if not self.face_db:
            print(f"[{self.text['info']}] {self.text['no_persons']}")
            return
            
        name = input(f"{self.text['delete_prompt']}").strip()
        if name in self.face_db:
            del self.face_db[name]
            self.save_database()
            
            # Delete face images
            face_dir = Path(f"./data/secure_faces/{name}")
            if face_dir.exists():
                import shutil
                shutil.rmtree(face_dir)
                
            print(f"[{self.text['success']}] {self.text['person_deleted']}: {name}")
        else:
            print(f"[{self.text['error']}] {self.text['person_not_found']}")

    def backup_database(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = f"backups/facegate_backup_{timestamp}.pkl"
        
        encrypted_db = {}
        for name, data in self.face_db.items():
            encrypted_db[name] = self.simple_encrypt(data)
        
        with open(backup_file, 'wb') as f:
            pickle.dump(encrypted_db, f)
            
        print(f"[{self.text['success']}] {self.text['backup_created']}: {backup_file}")

    def restore_database(self):
        backups = list(Path("backups").glob("facegate_backup_*.pkl"))
        if not backups:
            print(f"[{self.text['error']}] No backups found")
            return
            
        print(f"\nAvailable backups:")
        for i, backup in enumerate(backups, 1):
            print(f"  {i}. {backup.name}")
            
        try:
            choice = int(input("Select backup: ")) - 1
            if 0 <= choice < len(backups):
                with open(backups[choice], 'rb') as f:
                    encrypted_db = pickle.load(f)
                
                self.face_db = {}
                for name, encrypted_data in encrypted_db.items():
                    decrypted_data = self.simple_decrypt(encrypted_data)
                    if decrypted_data is not None:
                        self.face_db[name] = decrypted_data
                
                self.save_database()
                print(f"[{self.text['success']}] {self.text['restore_complete']}")
            else:
                print(f"[{self.text['error']}] {self.text['invalid_choice']}")
        except (ValueError, IndexError):
            print(f"[{self.text['error']}] {self.text['invalid_choice']}")

    def show_database_stats(self):
        print(f"\n[{self.text['database_stats']}]")
        total_persons = len(self.face_db)
        total_samples = sum(data['sample_count'] for data in self.face_db.values())
        
        db_size = 0
        for face_dir in Path("./data/secure_faces").glob("*"):
            if face_dir.is_dir():
                for img_file in face_dir.glob("*.jpg"):
                    db_size += img_file.stat().st_size
        
        print(f"  {self.text['total_persons']}: {total_persons}")
        print(f"  {self.text['total_samples']}: {total_samples}")
        print(f"  {self.text['db_size']}: {db_size / 1024 / 1024:.2f} MB")

    def language_selection(self):
        while True:
            print(f"\n[{self.text['language_title']}]")
            print("=" * 50)
            print(f"{self.text['current_language']}: {self.current_language.upper()}")
            print(f"{self.text['select_language']}:")
            for key, value in self.text['language_options'].items():
                print(f"  {key}. {value}")
            print("=" * 50)
            
            choice = input(f"{self.text['select_option']}").strip()
            
            if choice == '1':
                if self.change_language('hungarian'):
                    print(f"[{self.text['success']}] {self.text['language_changed']}: Magyar")
                break
            elif choice == '2':
                if self.change_language('english'):
                    print(f"[{self.text['success']}] {self.text['language_changed']}: English")
                break
            else:
                print(f"[{self.text['error']}] {self.text['invalid_choice']}")

    def main_menu(self):
        self.initialize_system()
        
        while True:
            print(f"\n[{self.text['main_menu_title']}]")
            print("=" * 70)
            print(f"Fejleszto: {self.system_info['developer']}")
            print(f"Projekt: {self.system_info['project']}")
            print(f"Platform: {self.platform.upper()} | Nyelv: {self.current_language.upper()}")
            print("=" * 70)
            
            for key, value in self.text['menu_options'].items():
                print(f"  {key}. {value}")
                
            print("=" * 70)
            
            choice = input(f"{self.text['select_option']}").strip()
            
            if choice == '1':
                self.register_face()
            elif choice == '2':
                self.security_system()
            elif choice == '3':
                self.settings_menu()
            elif choice == '4':
                self.show_system_information()
            elif choice == '5':
                self.database_management()
            elif choice == '6':
                self.language_selection()
            elif choice == '7':
                print(f"\n[{self.text['system_shutdown']}]")
                if self.arduino:
                    self.arduino.close()
                break
            else:
                print(f"[{self.text['error']}] {self.text['invalid_choice']}")
                
    def load_lbph_model(self):
        if self.lbph_recognizer is None:
            return
            
        try:
            self.lbph_recognizer.read('models/secure_facegate_lbph_model.xml')
            self.lbph_trained = True
            print(f"[{self.text['success']}] LBPH model loaded")
        except:
            print(f"[{self.text['info']}] No LBPH model found")
            self.lbph_trained = False

    def numpy_to_serializable(self, obj):
        if isinstance(obj, np.ndarray):
            return {
                '__numpy__': True,
                'dtype': str(obj.dtype),
                'data': base64.b64encode(obj.tobytes()).decode('utf-8'),
                'shape': obj.shape
            }
        elif isinstance(obj, (list, tuple)):
            return [self.numpy_to_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: self.numpy_to_serializable(value) for key, value in obj.items()}
        else:
            return obj
    
    def serializable_to_numpy(self, obj):
        if isinstance(obj, dict) and '__numpy__' in obj:
            data = base64.b64decode(obj['data'])
            return np.frombuffer(data, dtype=obj['dtype']).reshape(obj['shape'])
        elif isinstance(obj, list):
            return [self.serializable_to_numpy(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: self.serializable_to_numpy(value) for key, value in obj.items()}
        else:
            return obj
    
    def simple_encrypt(self, data):
        serializable_data = self.numpy_to_serializable(data)
        json_data = json.dumps(serializable_data).encode()
        
        key_extended = self.encryption_key * (len(json_data) // len(self.encryption_key) + 1)
        encrypted = bytes([json_data[i] ^ key_extended[i] for i in range(len(json_data))])
        
        return base64.b64encode(encrypted).decode('utf-8')
    
    def simple_decrypt(self, encrypted_data):
        try:
            encrypted_bytes = base64.b64decode(encrypted_data)
            key_extended = self.encryption_key * (len(encrypted_bytes) // len(self.encryption_key) + 1)
            decrypted = bytes([encrypted_bytes[i] ^ key_extended[i] for i in range(len(encrypted_bytes))])
            
            serializable_data = json.loads(decrypted.decode())
            return self.serializable_to_numpy(serializable_data)
            
        except Exception as e:
            self.logger.error(f"Decryption error: {e}")
            return None

if __name__ == "__main__":
    system = CrossPlatformFaceGate()
    system.main_menu()