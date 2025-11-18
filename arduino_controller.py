import serial
import serial.tools.list_ports
import time
import logging
import threading

class ArduinoController:
    def __init__(self, config):
        self.config = config
        self.serial_connection = None
        self.is_connected = False
        self.current_state = "ZARVA"
        self.lock_timer = None
        self.unlock_start_time = 0
        self.setup_logging()
        
    def setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("ArduinoController")
    
    def connect(self):
        try:
            port = self.find_arduino_port()
            if not port:
                self.logger.warning("Arduino nem talalhato. Szimulacios mod.")
                return False
            
            self.serial_connection = serial.Serial(
                port=port,
                baudrate=9600,
                timeout=1
            )
            
            time.sleep(2)
            self.is_connected = True
            self.logger.info(f"Csatlakozva Arduino-hoz: {port}")
            self.send_command("LOCK")
            return True
            
        except Exception as e:
            self.logger.error(f"Arduino csatlakozasi hiba: {e}")
            return False
    
    def find_arduino_port(self):
        try:
            ports = serial.tools.list_ports.comports()
            for port in ports:
                if any(keyword in port.description.lower() for keyword in ['arduino', 'ch340']):
                    return port.device
            return None
        except:
            return None
    
    def send_command(self, command):
        if self.is_connected and self.serial_connection:
            try:
                self.serial_connection.write(f"{command}\n".encode())
                return True
            except Exception as e:
                self.logger.error(f"Parancs kuldesi hiba: {e}")
                self.is_connected = False
                return False
        else:
            self.logger.info(f"[SZIMULACIO] Parancs: {command}")
            return True
    
    def unlock_door(self):
        if self.send_command("UNLOCK"):
            self.current_state = "NYITVA"
            self.unlock_start_time = time.time()
            
            if self.lock_timer and self.lock_timer.is_alive():
                self.lock_timer.cancel()
            
            self.lock_timer = threading.Timer(self.config.UNLOCK_DURATION, self.lock_door)
            self.lock_timer.start()
            return True
        return False
    
    def lock_door(self):
        if self.send_command("LOCK"):
            self.current_state = "ZARVA"
            return True
        return False
    
    def force_lock_door(self):
        if self.lock_timer and self.lock_timer.is_alive():
            self.lock_timer.cancel()
        
        if self.current_state == "NYITVA":
            self.lock_door()
    
    def get_unlock_remaining_time(self):
        if self.current_state == "NYITVA":
            elapsed = time.time() - self.unlock_start_time
            remaining = max(0, self.config.UNLOCK_DURATION - elapsed)
            return remaining
        return 0
    
    def disconnect(self):
        if self.lock_timer and self.lock_timer.is_alive():
            self.lock_timer.cancel()
        
        if self.is_connected and self.serial_connection:
            self.lock_door()
            time.sleep(1)
            self.serial_connection.close()
            self.is_connected = False