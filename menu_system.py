import cv2
import numpy as np

class MenuSystem:
    def __init__(self, face_recognizer, arduino_controller, config):
        self.face_recognizer = face_recognizer
        self.arduino = arduino_controller
        self.config = config
        self.current_menu = "main"
        self.selected_option = 0
        self.editing_value = False
        self.text_input = ""
        self.input_cursor = False
        self.cursor_timer = 0
        
        # Menu struktura
        self.menu_options = {
            "main": ["Arc Felismeres Inditasa", "Uj Arc Tanulasa", "Ismert Arcok", "Beallitasok", "Kilepes"],
            "learn": ["Nev Beirasa", "Tanulas Inditasa", "Vissza"],
            "manage": ["Ismert Arcok Listaja", "Arc Torlese", "Vissza"],
            "settings": [
                f"Biztonsagi Küszob: {self.config.CONFIDENCE_THRESHOLD:.2f}",
                f"Tanulo Mintak: {self.config.AUTO_LEARN_SAMPLES}",
                f"Nyitva Tartas: {self.config.UNLOCK_DURATION}s",
                "Vissza"
            ],
            "delete": []
        }
        
        # Szinek
        self.colors = {
            "background": (20, 20, 40),
            "header": (30, 30, 60),
            "option": (40, 40, 80),
            "selected": (70, 130, 180),
            "text": (220, 220, 220),
            "highlight": (50, 180, 255),
            "warning": (220, 80, 60),
            "success": (80, 200, 120)
        }

    def draw_menu(self, frame):
        height, width = frame.shape[:2]
        
        # Atlatszo menu hatter
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, height), self.colors["background"], -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        
        # Fejlec
        header_height = 80
        cv2.rectangle(frame, (0, 0), (width, header_height), self.colors["header"], -1)
        cv2.rectangle(frame, (0, header_height), (width, header_height + 2), self.colors["highlight"], -1)
        
        # Cim
        title = "FaceGate Biztonsagi Rendszer"
        title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
        cv2.putText(frame, title, (width//2 - title_size[0]//2, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, self.colors["text"], 3)
        
        # Menu tartalom
        content_y = header_height + 40
        self.draw_menu_content(frame, content_y, width, height)
        
        # Status bar
        self.draw_status_bar(frame, width, height)

    def draw_menu_content(self, frame, start_y, width, height):
        options = self.menu_options[self.current_menu]
        
        # Specialis menu kezeles
        if self.current_menu == "delete":
            self.draw_delete_menu(frame, start_y, width, height)
            return
        
        # Alap menu opciok
        option_height = 50
        menu_width = 400
        menu_x = width//2 - menu_width//2
        
        for i, option in enumerate(options):
            option_y = start_y + i * option_height
            
            # Opcio hatter
            color = self.colors["selected"] if i == self.selected_option else self.colors["option"]
            cv2.rectangle(frame, (menu_x, option_y), (menu_x + menu_width, option_y + 40), color, -1)
            cv2.rectangle(frame, (menu_x, option_y), (menu_x + menu_width, option_y + 40), self.colors["highlight"], 1)
            
            # Opcio szoveg
            text_color = self.colors["text"]
            if self.editing_value and i == self.selected_option:
                option_text = self.get_editing_text(option)
                text_color = self.colors["highlight"]
            else:
                option_text = option
                
            text_size = cv2.getTextSize(option_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            text_x = menu_x + 20
            text_y = option_y + 25
            
            cv2.putText(frame, option_text, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)

    def draw_delete_menu(self, frame, start_y, width, height):
        menu_width = 400
        menu_x = width//2 - menu_width//2
        
        # Cim
        cv2.putText(frame, "Arc Torlese", (menu_x, start_y - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.colors["text"], 2)
        
        known_persons = self.face_recognizer.get_known_persons()
        if not known_persons:
            cv2.putText(frame, "Nincs elmentett arc", (menu_x, start_y + 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors["text"], 1)
            return
        
        option_height = 40
        for i, person in enumerate(known_persons):
            option_y = start_y + i * option_height
            
            color = self.colors["selected"] if i == self.selected_option else self.colors["option"]
            cv2.rectangle(frame, (menu_x, option_y), (menu_x + menu_width, option_y + 35), color, -1)
            
            cv2.putText(frame, person, (menu_x + 20, option_y + 23), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors["text"], 1)
            
            # Torles gomb
            cv2.putText(frame, "[TORLES]", (menu_x + menu_width - 100, option_y + 23), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors["warning"], 1)

    def draw_status_bar(self, frame, width, height):
        status_y = height - 30
        
        # Status hatter
        cv2.rectangle(frame, (0, status_y), (width, height), self.colors["header"], -1)
        
        # Informaciok
        known_count = len(self.face_recognizer.get_known_persons())
        status_text = f"Ismert arcok: {known_count} | Ajto: {self.arduino.current_state}"
        
        cv2.putText(frame, status_text, (20, height - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors["text"], 1)
        
        # Utasitasok - VISSZA ALLITVA WASD-re
        instructions = "WASD: Navigalas | ENTER: Valasztas | ESC: Vissza/Megszakitas"
        instr_size = cv2.getTextSize(instructions, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
        cv2.putText(frame, instructions, (width - instr_size[0] - 20, height - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors["text"], 1)

    def get_editing_text(self, original_text):
        # Szoveg szerkesztes megjelenitese
        self.cursor_timer += 1
        cursor = "|" if self.cursor_timer % 60 < 30 else " "
        
        if self.current_menu == "learn" and self.selected_option == 0:
            return f"Nev: {self.text_input}{cursor}"
        elif self.current_menu == "settings":
            if self.selected_option == 0:
                return f"Biztonsagi Küszob: {self.text_input}{cursor}"
            elif self.selected_option == 1:
                return f"Tanulo Mintak: {self.text_input}{cursor}"
            elif self.selected_option == 2:
                return f"Nyitva Tartas: {self.text_input}{cursor}"
        
        return original_text

    def handle_input(self, key):
        if self.editing_value:
            return self.handle_text_input(key)
        else:
            return self.handle_navigation(key)

    def handle_navigation(self, key):
        options = self.menu_options[self.current_menu]
        
        # WASD navigalas - VISSZA ALLITVA
        if key == ord('w'):  # W - FEL
            self.selected_option = (self.selected_option - 1) % len(options)
            return "navigalas"
        elif key == ord('s'):  # S - LE
            self.selected_option = (self.selected_option + 1) % len(options)
            return "navigalas"
        elif key == 13:  # ENTER
            return self.select_option()
        elif key == 27:  # ESC
            return self.handle_escape()
        
        return None

    def handle_text_input(self, key):
        if key == 13:  # ENTER - befejezes
            self.finish_editing()
            return "szerkesztes_vege"
        elif key == 27:  # ESC - megszakitas
            self.cancel_editing()
            return "szerkesztes_megszakitas"
        elif key == 8:  # BACKSPACE
            self.text_input = self.text_input[:-1]
            return "szerkesztes"
        elif 32 <= key <= 126:  # Nyomtathato karakterek
            self.text_input += chr(key)
            return "szerkesztes"
        
        return None

    def handle_escape(self):
        if self.current_menu == "main":
            return "kilepes"
        else:
            self.current_menu = "main"
            self.selected_option = 0
            return "fo_menu"

    def select_option(self):
        options = self.menu_options[self.current_menu]
        selected = options[self.selected_option]
        
        if self.current_menu == "main":
            if selected == "Arc Felismeres Inditasa":
                return "arc_felismeres"
            elif selected == "Uj Arc Tanulasa":
                self.current_menu = "learn"
                self.selected_option = 0
                return "tanulas_menu"
            elif selected == "Ismert Arcok":
                self.current_menu = "manage"
                self.selected_option = 0
                return "kezeles_menu"
            elif selected == "Beallitasok":
                self.current_menu = "settings"
                self.selected_option = 0
                self.update_settings_display()
                return "beallitasok_menu"
            elif selected == "Kilepes":
                return "kilepes"
        
        elif self.current_menu == "learn":
            if selected == "Nev Beirasa":
                self.start_text_input("")
                return "nev_beiras"
            elif selected == "Tanulas Inditasa":
                if hasattr(self, 'learning_name') and self.learning_name:
                    return "tanulas_inditas"
                else:
                    return "nev_előbb"
            elif selected == "Vissza":
                self.current_menu = "main"
                self.selected_option = 0
                return "fo_menu"
        
        elif self.current_menu == "manage":
            if selected == "Ismert Arcok Listaja":
                return "arcok_listazasa"
            elif selected == "Arc Torlese":
                self.current_menu = "delete"
                self.selected_option = 0
                self.menu_options["delete"] = self.face_recognizer.get_known_persons()
                return "torles_menu"
            elif selected == "Vissza":
                self.current_menu = "main"
                self.selected_option = 0
                return "fo_menu"
        
        elif self.current_menu == "settings":
            if selected.startswith("Biztonsagi Küszob:"):
                self.start_text_input(str(self.config.CONFIDENCE_THRESHOLD))
                return "küszob_szerkesztes"
            elif selected.startswith("Tanulo Mintak:"):
                self.start_text_input(str(self.config.AUTO_LEARN_SAMPLES))
                return "mintak_szerkesztes"
            elif selected.startswith("Nyitva Tartas:"):
                self.start_text_input(str(self.config.UNLOCK_DURATION))
                return "tartas_szerkesztes"
            elif selected == "Vissza":
                self.current_menu = "main"
                self.selected_option = 0
                return "fo_menu"
        
        elif self.current_menu == "delete":
            known_persons = self.face_recognizer.get_known_persons()
            if known_persons and self.selected_option < len(known_persons):
                person_to_delete = known_persons[self.selected_option]
                self.face_recognizer.delete_person(person_to_delete)
                self.menu_options["delete"] = self.face_recognizer.get_known_persons()
                if not self.menu_options["delete"]:
                    self.current_menu = "manage"
                return "arc_torolve"
        
        return None

    def start_text_input(self, initial_text):
        self.editing_value = True
        self.text_input = initial_text
        self.cursor_timer = 0

    def finish_editing(self):
        self.editing_value = False
        
        if self.current_menu == "learn" and self.selected_option == 0:
            self.learning_name = self.text_input.strip()
        elif self.current_menu == "settings":
            try:
                if self.selected_option == 0:  # Biztonsagi küszob
                    value = float(self.text_input)
                    if 0.1 <= value <= 1.0:
                        self.config.CONFIDENCE_THRESHOLD = value
                        self.face_recognizer.confidence_threshold = value
                elif self.selected_option == 1:  # Tanulo mintak
                    value = int(self.text_input)
                    if 5 <= value <= 50:
                        self.config.AUTO_LEARN_SAMPLES = value
                        self.face_recognizer.target_samples = value
                elif self.selected_option == 2:  # Nyitva tartas
                    value = int(self.text_input)
                    if 1 <= value <= 60:
                        self.config.UNLOCK_DURATION = value
            except ValueError:
                pass
            
            self.config.save_config()
            self.update_settings_display()
        
        self.text_input = ""

    def cancel_editing(self):
        self.editing_value = False
        self.text_input = ""

    def update_settings_display(self):
        self.menu_options["settings"] = [
            f"Biztonsagi Küszob: {self.config.CONFIDENCE_THRESHOLD:.2f}",
            f"Tanulo Mintak: {self.config.AUTO_LEARN_SAMPLES}",
            f"Nyitva Tartas: {self.config.UNLOCK_DURATION}s",
            "Vissza"
        ]

    def get_learning_name(self):
        return getattr(self, 'learning_name', '')