import torch
import torch.nn as nn
import cv2
import numpy as np
import os
import pickle
import logging
import time

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 20 * 20, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.classifier(x)
        return x

class NeuralFaceRecognizer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.setup_logging()
        
        # Arc detektor
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # CNN modell
        self.model = None
        self.face_features = {}
        self.confidence_threshold = config.CONFIDENCE_THRESHOLD
        
        # Automatikus tanulas
        self.learning_mode = False
        self.current_learning_name = ""
        self.learned_samples = 0
        self.target_samples = config.AUTO_LEARN_SAMPLES
        self.last_learn_time = 0
        
        self.load_model()
        
    def setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("NeuralFaceRecognizer")
    
    def load_model(self):
        model_path = "face_model.pth"
        if os.path.exists(model_path):
            self.model = SimpleCNN(num_classes=100)
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
        
        faces_path = "known_faces.pkl"
        if os.path.exists(faces_path):
            with open(faces_path, 'rb') as f:
                data = pickle.load(f)
                self.face_features = data.get('face_features', {})
    
    def save_model(self):
        if self.model:
            torch.save(self.model.state_dict(), "face_model.pth")
        
        data = {
            'face_features': self.face_features
        }
        with open("known_faces.pkl", 'wb') as f:
            pickle.dump(data, f)
    
    def detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        face_locations = []
        face_images = []
        
        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            if face_img.size > 0:
                face_img = cv2.resize(face_img, (160, 160))
                face_images.append(face_img)
                face_locations.append((y, x+w, y+h, x))
        
        return face_locations, face_images
    
    def extract_features(self, face_image):
        if not self.model:
            hist = cv2.calcHist([face_image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            return cv2.normalize(hist, hist).flatten()
            
        face_tensor = self.preprocess_face(face_image)
        
        with torch.no_grad():
            features = self.model.conv_layers(face_tensor)
            features = features.cpu().numpy().flatten()
        
        return features
    
    def preprocess_face(self, face_image):
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        face_image = face_image.astype(np.float32) / 255.0
        face_tensor = torch.from_numpy(face_image).permute(2, 0, 1).unsqueeze(0)
        return face_tensor.to(self.device)
    
    def recognize_face(self, face_image):
        if len(self.face_features) == 0:
            return "ISmeretlen", 0.0
        
        features = self.extract_features(face_image)
        
        best_match = "ISmeretlen"
        best_confidence = 0.0
        
        for name, known_features_list in self.face_features.items():
            for known_features in known_features_list:
                similarity = self.calculate_similarity(features, known_features)
                if similarity > best_confidence:
                    best_confidence = similarity
                    best_match = name
        
        return best_match, best_confidence
    
    def calculate_similarity(self, features1, features2):
        dot_product = np.dot(features1, features2)
        norm1 = np.linalg.norm(features1)
        norm2 = np.linalg.norm(features2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return dot_product / (norm1 * norm2)
    
    def start_learning(self, name):
        if name not in self.face_features:
            self.face_features[name] = []
        
        self.learning_mode = True
        self.current_learning_name = name
        self.learned_samples = 0
        self.last_learn_time = time.time()
    
    def add_learning_sample(self, face_image):
        if not self.learning_mode:
            return False
        
        current_time = time.time()
        if current_time - self.last_learn_time < 0.5:
            return True
        
        if self.learned_samples >= self.target_samples:
            self.stop_learning()
            return False
        
        features = self.extract_features(face_image)
        self.face_features[self.current_learning_name].append(features)
        self.learned_samples += 1
        self.last_learn_time = current_time
        
        if self.learned_samples >= self.target_samples:
            self.stop_learning()
            return False
        
        return True
    
    def stop_learning(self):
        if self.learning_mode:
            self.save_model()
            self.learning_mode = False
            self.current_learning_name = ""
    
    def get_learning_progress(self):
        return self.learned_samples, self.target_samples
    
    def is_learning(self):
        return self.learning_mode
    
    def get_known_persons(self):
        return list(self.face_features.keys())
    
    def delete_person(self, name):
        if name in self.face_features:
            del self.face_features[name]
            self.save_model()
            return True
        return False
    
    def should_unlock(self, recognized_names, confidences):
        if not recognized_names:
            return False
        
        all_known = all(name != "ISmeretlen" for name in recognized_names)
        all_confident = all(conf >= self.confidence_threshold for conf in confidences)
        
        return all_known and all_confident and len(recognized_names) > 0