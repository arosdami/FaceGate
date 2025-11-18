import cv2
import logging
import time
import numpy as np
from config import Config
from arduino_controller import ArduinoController
from neural_face_recognizer import NeuralFaceRecognizer
from menu_system import MenuSystem

config = Config()

class FaceGateSystem:
    def __init__(self):
        self.setup_logging()
        self.face_recognizer = NeuralFaceRecognizer(config)
        self.arduino = ArduinoController(config)
        self.menu_system = MenuSystem(self.face_recognizer, self.arduino, config)
        self.camera = None
        self.frame_count = 0
        self.system_mode = "menu"
        
    def setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("FaceGateSystem")
    
    def initialize_camera(self):
        try:
            self.camera = cv2.VideoCapture(config.CAMERA_ID)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
            
            if not self.camera.isOpened():
                self.logger.error("Kamera nem nyithato meg")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Kamera hiba: {e}")
            return False
    
    def draw_recognition_ui(self, frame, face_locations, names, confidences):
        # Arc detekciok
        for (top, right, bottom, left), name, confidence in zip(face_locations, names, confidences):
            if name != "ISmeretlen" and confidence >= self.face_recognizer.confidence_threshold:
                color = (0, 255, 0)  # Zold - ismert
                status_text = name
            else:
                color = (0, 0, 255)  # Piros - ismeretlen
                status_text = "ISMERETLEN"
            
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            
            status = f"{status_text} ({confidence:.2f})"
            cv2.putText(frame, status, (left + 6, bottom - 6),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Status info
        known_count = sum(1 for name in names if name != "ISmeretlen" and any(conf >= self.face_recognizer.confidence_threshold for conf in confidences))
        unknown_count = len(names) - known_count
        
        door_color = (0, 255, 0) if self.arduino.current_state == "NYITVA" else (0, 0, 255)
        cv2.putText(frame, f"AJTO: {self.arduino.current_state}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, door_color, 2)
        
        if self.arduino.current_state == "NYITVA":
            remaining = self.arduino.get_unlock_remaining_time()
            cv2.putText(frame, f"Zaras: {remaining:.1f}s", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.putText(frame, f"ISMERT: {known_count} | ISMERETLEN: {unknown_count}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.putText(frame, "M: Menu | ESC: Kilepes", (10, config.FRAME_HEIGHT - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def draw_learning_ui(self, frame, face_locations):
        for (top, right, bottom, left) in face_locations:
            cv2.rectangle(frame, (left, top), (right, bottom), (255, 255, 0), 2)
        
        progress, target = self.face_recognizer.get_learning_progress()
        learning_name = self.menu_system.get_learning_name()
        
        cv2.putText(frame, f"TANULAS: {learning_name}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"Haladas: {progress}/{target}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, "Nezz a kameraba!", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, "ESC: Megszakitas", (10, config.FRAME_HEIGHT - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def run_recognition(self, frame):
        face_locations, faces = self.face_recognizer.detect_faces(frame)
        
        recognized_names = []
        confidences = []
        
        for face in faces:
            name, confidence = self.face_recognizer.recognize_face(face)
            recognized_names.append(name)
            confidences.append(confidence)
        
        # Biztonsagi logika
        should_unlock = self.face_recognizer.should_unlock(recognized_names, confidences)
        
        if should_unlock:
            if self.arduino.current_state == "ZARVA":
                self.arduino.unlock_door()
        else:
            if any(name == "ISmeretlen" for name in recognized_names):
                self.arduino.force_lock_door()
        
        self.draw_recognition_ui(frame, face_locations, recognized_names, confidences)
    
    def run_learning(self, frame):
        face_locations, faces = self.face_recognizer.detect_faces(frame)
        
        if faces and self.face_recognizer.is_learning():
            self.face_recognizer.add_learning_sample(faces[0])
        
        self.draw_learning_ui(frame, face_locations)
        
        if not self.face_recognizer.is_learning():
            self.system_mode = "menu"
    
    def run(self):
        self.logger.info("FaceGate rendszer inditasa...")
        
        self.arduino.connect()
        
        if not self.initialize_camera():
            return
        
        self.logger.info("Rendszer kesz. M billentyu a menuhoz.")
        
        try:
            while True:
                ret, frame = self.camera.read()
                if not ret:
                    break
                
                self.frame_count += 1
                if self.frame_count % config.FRAME_SKIP != 0:
                    continue
                
                # Mod kezeles
                if self.system_mode == "menu":
                    self.menu_system.draw_menu(frame)
                elif self.system_mode == "arc_felismeres":
                    self.run_recognition(frame)
                elif self.system_mode == "tanulas":
                    self.run_learning(frame)
                
                cv2.imshow(config.WINDOW_NAME, frame)
                
                key = cv2.waitKey(1)
                if key == -1:
                    continue
                
                # Billentyu kezeles - VISSZA ALLITVA WASD-re
                if key == config.EXIT_KEY:
                    break
                elif key == ord('m') or key == ord('M'):
                    self.system_mode = "menu"
                    print("Menu mod")
                
                # Menu input kezeles
                if self.system_mode == "menu":
                    action = self.menu_system.handle_input(key)
                    if action == "arc_felismeres":
                        self.system_mode = "arc_felismeres"
                        print("Arc felismeres inditva")
                    elif action == "tanulas_inditas":
                        learning_name = self.menu_system.get_learning_name()
                        if learning_name:
                            self.face_recognizer.start_learning(learning_name)
                            self.system_mode = "tanulas"
                            print(f"Tanulas inditva: {learning_name}")
                        else:
                            print("Tanulas megszakítva - nincs nev")
                    elif action == "kilepes":
                        break
                    elif action == "navigalas":
                        pass  # Csak navigalt
                    
        except KeyboardInterrupt:
            self.logger.info("Felhasznalo megszakította")
        except Exception as e:
            self.logger.error(f"Rendszer hiba: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        self.logger.info("Rendszer leallitasa...")
        
        if self.camera:
            self.camera.release()
        
        self.arduino.disconnect()
        cv2.destroyAllWindows()
        
        self.logger.info("Rendszer leallitva")

if __name__ == "__main__":
    system = FaceGateSystem()
    system.run()
