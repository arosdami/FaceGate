import os
import torch
import json

class Config:
    # GPU beallitasok
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Data mappa beallitasai
    DATA_PATH = "data"
    MODEL_PATH = "models"
    CONFIG_FILE = "system_config.json"
    ALLOWED_EXTENSIONS = ('.jpg', '.jpeg', '.png')
    
    # Kamera beallitasok
    CAMERA_ID = 0
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480
    FRAME_SKIP = 2
    
    # Arc detekcio
    FACE_DETECTION_CONFIDENCE = 0.7
    MIN_FACE_SIZE = 30
    
    # Alap ertekek - ezek feluliródnak a betöltött konfiggal
    CONFIDENCE_THRESHOLD = 0.95
    AUTO_LEARN_SAMPLES = 50
    UNLOCK_DURATION = 10
    
    # Arduino beallitasok
    ARDUINO_PORT = "COM4"
    ARDUINO_BAUDRATE = 9600
    ARDUINO_TIMEOUT = 1
    
    # Alkalmazas beallitasok
    WINDOW_NAME = "FaceGate Biztonsagi Rendszer"
    EXIT_KEY = 27

    def __init__(self):
        self.load_config()

    def load_config(self):
        if os.path.exists(self.CONFIG_FILE):
            try:
                with open(self.CONFIG_FILE, 'r') as f:
                    config_data = json.load(f)
                    self.CONFIDENCE_THRESHOLD = config_data.get('confidence_threshold', 0.75)
                    self.AUTO_LEARN_SAMPLES = config_data.get('auto_learn_samples', 15)
                    self.UNLOCK_DURATION = config_data.get('unlock_duration', 10)
            except:
                pass

    def save_config(self):
        config_data = {
            'confidence_threshold': self.CONFIDENCE_THRESHOLD,
            'auto_learn_samples': self.AUTO_LEARN_SAMPLES,
            'unlock_duration': self.UNLOCK_DURATION
        }
        with open(self.CONFIG_FILE, 'w') as f:
            json.dump(config_data, f, indent=4)

config = Config()