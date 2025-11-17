import cv2
import numpy as np
import pickle
import os
from pathlib import Path
import time
import serial
import serial.tools.list_ports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import json
from datetime import datetime
import sys

class CNNArcFelismero:
    def __init__(self, input_shape=(128, 128, 3), num_classes=2):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self._epites_modell()
        
    def _epites_modell(self):
        model = keras.Sequential([
            # Elso konvolucios blokk
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Masodik konvolucios blokk
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Harmadik konvolucios blokk
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Negyedik konvolucios blokk
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Besimitas es teljesen osszekapcsolt retegek
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def tanitas(self, X_train, y_train, X_val=None, y_val=None, epochs=50, batch_size=32):
        callbacks = [
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
        ]
        
        validation_data = (X_val, y_val) if X_val is not None else None
        
        historia = self.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )
        
        return historia
    
    def predikalas(self, kep):
        if len(kep.shape) == 3:
            kep = np.expand_dims(kep, axis=0)
        return self.model.predict(kep)
    
    def mentes(self, fajl_ut):
        self.model.save(fajl_ut)
    
    def betoltes(self, fajl_ut):
        self.model = keras.models.load_model(fajl_ut)

class BiztonsagiArcRendszer:
    def __init__(self):
        # Arc- es szembesorolok betoltese
        self.arc_besorolo = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.szem_besorolo = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # CNN modell inicializalasa
        self.cnn_modell = CNNArcFelismero()
        self.modell_betoltve = False
        
        # Adatbazisok inicializalasa
        self.jovahagyott_adatbazis = {}
        self.tiltott_adatbazis = {}
        self.osztaly_nev_kodolas = {}
        self.kovetkezo_osztaly_kod = 0
        
        # Rendszer beallitasok
        self.beallitasok = {
            'kuszob': 0.85,
            'nyitva_tartas': 8,
            'kamera_index': 0,
            'arduino_port': None,
            'max_mintak': 50,
            'tanitas_epochs': 30,
            'batch_size': 16,
            'auto_capture_delay': 0.5  # Automatikus rogzites kozotti kesleltetes
        }
        
        self.serial_connection = None
        self.konyvtarak_beallitasa()
        self.adatbazisok_betoltese()
        self.modell_betoltese()
        
        # Rendszer informaciok
        self.rendszer_info = {
            'developer': 'Damjan Aros - THE PTI',
            'project': 'Egyetemi Projekt - FaceGate',
            'version': '1.1.0',
            'course': 'Robotika, MI & NN',
            'university': 'Tokaj-Hegyalja Egyetem',
            'supervisor': 'Attila Perlaki, David Gegeny',
            'semester': '2025 Os'
        }
        
    def typewriter_effect(self, text, delay=0.03):
        """Typewriter effect a szoveg kiirasahoz"""
        for char in text:
            print(char, end='', flush=True)
            time.sleep(delay)
        print()
    
    def konyvtarak_beallitasa(self):
        # Szukseges konyvtarak letrehozasa
        Path("data/jovahagyott").mkdir(parents=True, exist_ok=True)
        Path("data/tiltott").mkdir(parents=True, exist_ok=True)
        Path("models").mkdir(parents=True, exist_ok=True)
        Path("training_data").mkdir(parents=True, exist_ok=True)
        
    def adatbazisok_betoltese(self):
        # Jovahagyott es tiltott arcok adatbazisanak betoltese
        try:
            with open('data/jovahagyott_adatbazis.pkl', 'rb') as f:
                self.jovahagyott_adatbazis = pickle.load(f)
            print(f"OK Jovahagyott adatbazis betoltve: {len(self.jovahagyott_adatbazis)} szemely")
        except:
            print("INFO Nincs jovahagyott adatbazis")
            
        try:
            with open('data/tiltott_adatbazis.pkl', 'rb') as f:
                self.tiltott_adatbazis = pickle.load(f)
            print(f"OK Tiltott adatbazis betoltve: {len(self.tiltott_adatbazis)} szemely")
        except:
            print("INFO Nincs tiltott adatbazis")
            
        try:
            with open('data/osztaly_kodolas.pkl', 'rb') as f:
                self.osztaly_nev_kodolas = pickle.load(f)
                self.kovetkezo_osztaly_kod = max(self.osztaly_nev_kodolas.values()) + 1 if self.osztaly_nev_kodolas else 0
            print(f"OK Osztaly kodolas betoltve: {len(self.osztaly_nev_kodolas)} osztaly")
        except:
            print("INFO Nincs osztaly kodolas")
    
    def modell_betoltese(self):
        """CNN modell betoltese"""
        try:
            if os.path.exists("models/arc_felismero_model.h5"):
                self.cnn_modell.betoltes("models/arc_felismero_model.h5")
                self.modell_betoltve = True
                print("OK CNN modell sikeresen betoltve")
            else:
                print("INFO Nincs mentett CNN modell - uj modell inicializalva")
                self.modell_betoltve = False
        except Exception as e:
            print(f"ERROR Modell betoltesi hiba: {e}")
            self.modell_betoltve = False
    
    def modell_mentese(self):
        """CNN modell mentese"""
        try:
            self.cnn_modell.mentes("models/arc_felismero_model.h5")
            print("OK CNN modell sikeresen elmentve")
        except Exception as e:
            print(f"ERROR Modell mentesi hiba: {e}")
    
    def elerheto_portok_listazasa(self, tipus="mind"):
        """Elerheto portok listazasa"""
        portok = []
        
        if tipus in ["mind", "serial"]:
            # Serial portok listazasa
            elerheto_portok = serial.tools.list_ports.comports()
            for port in elerheto_portok:
                portok.append({
                    'tipus': 'serial',
                    'nev': port.device,
                    'leiras': f"{port.description} - {port.hwid}"
                })
        
        if tipus in ["mind", "kamera"]:
            # Kamera portok tesztelese
            for i in range(5):
                kamera = cv2.VideoCapture(i)
                if kamera.isOpened():
                    portok.append({
                        'tipus': 'kamera',
                        'nev': str(i),
                        'leiras': f"Kamera index {i}"
                    })
                    kamera.release()
        
        return portok

    def port_valasztas(self, tipus="kamera"):
        """Port kivalasztasa a felhasznalotol"""
        portok = self.elerheto_portok_listazasa(tipus)
        
        if not portok:
            print(f"ERROR Nincs elerheto {tipus} port!")
            return None
        
        print(f"\n{'='*40}")
        print(f"ELERHETO {tipus.upper()} PORTOK")
        print('='*40)
        for i, port in enumerate(portok):
            print(f"{i+1}. {port['nev']} - {port['leiras']}")
        
        try:
            valasztas = int(input(f"\nValassz {tipus} portot (1-{len(portok)}): ")) - 1
            if 0 <= valasztas < len(portok):
                kivalasztott_port = portok[valasztas]['nev']
                print(f"OK Kivalasztott port: {kivalasztott_port}")
                return kivalasztott_port
            else:
                print("ERROR Ervenytelen valasztas!")
                return None
        except ValueError:
            print("ERROR Ervenytelen bemenet!")
            return None

    def arduino_csatlakozas(self, port=None):
        """Arduino-hoz valo csatlakozas"""
        if port is None:
            port = self.port_valasztas("serial")
            if port is None:
                print("ERROR Nincs kivalasztva Arduino port!")
                return False
        
        try:
            self.serial_connection = serial.Serial(
                port=port,
                baudrate=9600,
                timeout=1
            )
            time.sleep(2)  # Varakozas az Arduino bootolasara
            print(f"OK Sikeresen csatlakozva az Arduino-hoz: {port}")
            
            # Arduino inicializalas ellenorzese
            if self.serial_connection.in_waiting > 0:
                init_uzi = self.serial_connection.readline().decode().strip()
                print(f"INFO Arduino uzenet: {init_uzi}")
            
            self.beallitasok['arduino_port'] = port
            return True
        except Exception as e:
            print(f"ERROR Arduino csatlakozasi hiba: {e}")
            self.serial_connection = None
            return False

    def zaras_vezerles(self, parancs):
        """Zaras vezerlese Arduino-n keresztul"""
        if self.serial_connection and self.serial_connection.is_open:
            try:
                if parancs == "NYITAS":
                    self.serial_connection.write(b"UNLOCK\n")
                    print("OK NYITAS parancs elkuldve az Arduino-nak")
                elif parancs == "ZARAS":
                    self.serial_connection.write(b"LOCK\n")
                    print("OK ZARAS parancs elkuldve az Arduino-nak")
                
                # Valasz olvasasa
                time.sleep(0.5)
                if self.serial_connection.in_waiting > 0:
                    valasz = self.serial_connection.readline().decode().strip()
                    print(f"INFO Arduino valasz: {valasz}")
                    
            except Exception as e:
                print(f"ERROR Arduino kommunikacios hiba: {e}")

    def arc_eszleles_es_analizis(self, kep):
        """Arc detektalas es minosegellenorzes"""
        szurke = cv2.cvtColor(kep, cv2.COLOR_BGR2GRAY)
        
        # Arcok detektalasa
        arcok = self.arc_besorolo.detectMultiScale(
            szurke, 
            scaleFactor=1.1, 
            minNeighbors=6, 
            minSize=(120, 120)
        )
        
        if len(arcok) == 0:
            return None, None, {}
            
        # Legnagyobb arc kivalasztasa
        x, y, w, h = max(arcok, key=lambda rect: rect[2] * rect[3])
        arc_reszlet = kep[y:y+h, x:x+w]
        
        # Arc minosegenek elemzese
        szurke_arc = cv2.cvtColor(arc_reszlet, cv2.COLOR_BGR2GRAY)
        fenyesseg = np.mean(szurke_arc)
        kontraszt = np.std(szurke_arc)
        
        # Szemek detektalasa minosegellenorzeshez
        szemek = self.szem_besorolo.detectMultiScale(szurke_arc, 1.1, 3)
        
        minoseg = "JO" if fenyesseg > 80 and kontraszt > 40 and len(szemek) >= 1 else "ROSSZ"
        
        analizis = {
            'minoseg': minoseg,
            'fenyesseg': fenyesseg,
            'kontraszt': kontraszt,
            'szemek_szama': len(szemek)
        }
        
        # Arc elofeldolgozasa CNN-hez
        arc_feldolgozva = cv2.resize(arc_reszlet, (128, 128))
        arc_feldolgozva = cv2.cvtColor(arc_feldolgozva, cv2.COLOR_BGR2RGB)
        arc_feldolgozva = arc_feldolgozva.astype('float32') / 255.0
        
        return arc_feldolgozva, (x, y, w, h), analizis

    def arc_felismeres_cnn(self, arc_kep):
        """Arc felismeres CNN modellel"""
        if not self.modell_betoltve:
            return "Ismeretlen", 0.0, "semmi"
        
        # Predikcio a CNN modellel
        predikcio = self.cnn_modell.predikalas(arc_kep)[0]
        
        # Legvaloszinuobb osztaly meghatarozasa
        osztaly_index = np.argmax(predikcio)
        bizonyossag = predikcio[osztaly_index]
        
        # Osztaly nevenek visszakeresese
        for nev, kod in self.osztaly_nev_kodolas.items():
            if kod == osztaly_index:
                if nev in self.jovahagyott_adatbazis:
                    return nev, bizonyossag, "jovahagyott"
                elif nev in self.tiltott_adatbazis:
                    return nev, bizonyossag, "tiltott"
        
        return "Ismeretlen", bizonyossag, "semmi"

    def adatbazis_elokeszitese_tanitashoz(self):
        """Adatbazis elokeszitese CNN tanitashoz"""
        X = []
        y = []
        
        # Jovahagyott arcok hozzaadasa
        for nev, adatok in self.jovahagyott_adatbazis.items():
            if nev not in self.osztaly_nev_kodolas:
                self.osztaly_nev_kodolas[nev] = self.kovetkezo_osztaly_kod
                self.kovetkezo_osztaly_kod += 1
            
            osztaly_kod = self.osztaly_nev_kodolas[nev]
            for kep in adatok['kepek']:
                X.append(kep)
                y.append(osztaly_kod)
        
        # Tiltott arcok hozzaadasa
        for nev, adatok in self.tiltott_adatbazis.items():
            if nev not in self.osztaly_nev_kodolas:
                self.osztaly_nev_kodolas[nev] = self.kovetkezo_osztaly_kod
                self.kovetkezo_osztaly_kod += 1
            
            osztaly_kod = self.osztaly_nev_kodolas[nev]
            for kep in adatok['kepek']:
                X.append(kep)
                y.append(osztaly_kod)
        
        if len(X) == 0:
            return None, None, None
        
        X = np.array(X)
        y = np.array(y)
        
        # One-hot encoding
        y_one_hot = keras.utils.to_categorical(y, num_classes=len(self.osztaly_nev_kodolas))
        
        return X, y_one_hot, len(self.osztaly_nev_kodolas)

    def cnn_tanitas(self):
        """CNN modell tanitasa"""
        print("\n" + "="*50)
        print("CNN MODELL TANITASA")
        print("="*50)
        
        # Adatok elokeszitese
        X, y, num_classes = self.adatbazis_elokeszitese_tanitashoz()
        
        if X is None:
            print("ERROR Nincsenek tanito adatok!")
            return False
        
        print(f"OK Tanito adatok betoltve:")
        print(f"  - Mintak: {X.shape[0]}")
        print(f"  - Osztalyok: {num_classes}")
        print(f"  - Epochs: {self.beallitasok['tanitas_epochs']}")
        
        # Uj modell epitese a megfelelo szamu osztallyal
        self.cnn_modell = CNNArcFelismero(num_classes=num_classes)
        
        # Adatok felosztasa tanito es validacios halmazra
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print(f"  - Tanito halmaz: {X_train.shape[0]} minta")
        print(f"  - Validacios halmaz: {X_val.shape[0]} minta")
        
        # Tanitas
        print("\nTANITAS ELINDITVA...")
        start_ido = time.time()
        
        try:
            historia = self.cnn_modell.tanitas(
                X_train, y_train, X_val, y_val,
                epochs=self.beallitasok['tanitas_epochs'],
                batch_size=self.beallitasok['batch_size']
            )
            
            tanitas_ido = time.time() - start_ido
            print(f"OK Tanitas befejezve: {tanitas_ido:.2f} masodperc")
            
            # Eredmenyek megjelenitese
            vegso_accuracy = historia.history['accuracy'][-1]
            vegso_val_accuracy = historia.history['val_accuracy'][-1] if 'val_accuracy' in historia.history else None
            
            print(f"\nTANITASI EREDMENYEK:")
            print(f"  - Vegso pontossag: {vegso_accuracy:.4f}")
            if vegso_val_accuracy:
                print(f"  - Validacios pontossag: {vegso_val_accuracy:.4f}")
            
            # Modell mentese
            self.modell_mentese()
            self.modell_betoltve = True
            
            # Osztaly kodolas mentese
            with open('data/osztaly_kodolas.pkl', 'wb') as f:
                pickle.dump(self.osztaly_nev_kodolas, f)
            
            return True
            
        except Exception as e:
            print(f"ERROR Hiba a tanitas soran: {e}")
            return False

    def arc_regisztralasa_auto(self, nev, adatbazis_tipus="jovahagyott"):
        """Uj arc regisztralasa automatikus adatgyujtessel"""
        print(f"\n{'='*40}")
        print(f"ARC REGISZTRALASA: {nev} ({adatbazis_tipus})")
        print('='*40)
        
        cel_adatbazis = self.jovahagyott_adatbazis if adatbazis_tipus == "jovahagyott" else self.tiltott_adatbazis
        cel_konyvtar = "jovahagyott" if adatbazis_tipus == "jovahagyott" else "tiltott"
        
        arc_konyvtar = Path(f"data/{cel_konyvtar}/{nev}")
        arc_konyvtar.mkdir(parents=True, exist_ok=True)
        
        kamera = cv2.VideoCapture(self.beallitasok['kamera_index'])
        if not kamera.isOpened():
            print(f"ERROR: Nem sikerult megnyitni a kamerat (index: {self.beallitasok['kamera_index']})")
            return False
            
        osszegyujtott_kepek = []
        gyujtott = 0
        utolso_capture_ido = 0
        
        print("Automatikus adatgyujtes inditva...")
        print("Nyomj 'q'-t a kilepeshez")
        
        try:
            while gyujtott < self.beallitasok['max_mintak']:
                ret, kep = kamera.read()
                if not ret:
                    continue
                    
                arc_kep, hatarolo, analizis = self.arc_eszleles_es_analizis(kep)
                megjelenito_kep = kep.copy()
                
                if arc_kep is not None:
                    x, y, w, h = hatarolo
                    
                    # Minoseg megjelenitese
                    minoseg_szin = (0, 255, 0) if analizis['minoseg'] == "JO" else (0, 0, 255)
                    minoseg_szoveg = f"MINOSEG: {analizis['minoseg']}"
                    
                    cv2.rectangle(megjelenito_kep, (x,y), (x+w,y+h), minoseg_szin, 3)
                    cv2.putText(megjelenito_kep, minoseg_szoveg, (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, minoseg_szin, 2)
                    
                    # Automatikus minta gyujtese jo minoseg eseten
                    aktualis_ido = time.time()
                    if (analizis['minoseg'] == "JO" and 
                        aktualis_ido - utolso_capture_ido > self.beallitasok['auto_capture_delay']):
                        
                        # Minta mentese
                        osszegyujtott_kepek.append(arc_kep)
                        
                        # Arc kep mentese fajlba
                        x, y, w, h = hatarolo
                        szurke_arc = cv2.cvtColor(kep[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
                        szurke_arc = cv2.resize(szurke_arc, (100, 100))
                        cv2.imwrite(str(arc_konyvtar / f"{gyujtott}.jpg"), szurke_arc)
                        
                        gyujtott += 1
                        utolso_capture_ido = aktualis_ido
                        print(f"OK Minta {gyujtott} automatikusan rogzitve")
                
                # Statisztikak megjelenitese
                cv2.putText(megjelenito_kep, f"GYUJTOTT: {gyujtott}/{self.beallitasok['max_mintak']}", 
                           (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(megjelenito_kep, f"MINOSEG: {analizis.get('minoseg', 'N/A')}", 
                           (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(megjelenito_kep, "AUTOMATIKUS ROGZITES AKTIV", 
                           (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                cv2.imshow("Arc Regisztracio - Automatikus", megjelenito_kep)
                
                billentyu = cv2.waitKey(1) & 0xFF
                if billentyu == ord('q'):
                    break
                    
        finally:
            kamera.release()
            cv2.destroyAllWindows()
        
        if len(osszegyujtott_kepek) >= 15:
            cel_adatbazis[nev] = {
                'kepek': osszegyujtott_kepek,
                'regisztracio_ido': time.time(),
                'mintak_szama': len(osszegyujtott_kepek),
                'adatbazis_tipus': adatbazis_tipus
            }
            
            self.adatbazis_mentese()
            print(f"OK Sikeres regisztracio: {nev} - {len(osszegyujtott_kepek)} minta")
            
            # CNN modell ujratanitasa
            print("\nCNN modell frissitese...")
            self.cnn_tanitas()
            
            return True
        else:
            print("ERROR Nincs elegendo jo minosegu minta (minimum 15 szukseges)")
            return False

    def biztonsagi_rendszer_inditasa(self):
        """Fo biztonsagi rendszer inditasa"""
        if not self.modell_betoltve:
            print("ERROR CNN modell nincs betoltve! Eloször tanitsd a modellt.")
            return
            
        print("\n" + "="*50)
        print("BIZTONSAGI RENDSZER INDITASA")
        print("="*50)
        print("Q - Kilepes, L - Zaras")
        
        # Kamera inicializalasa
        kamera = cv2.VideoCapture(self.beallitasok['kamera_index'])
        if not kamera.isOpened():
            print(f"ERROR: Nem sikerult megnyitni a kamerat (index: {self.beallitasok['kamera_index']})")
            return
        
        # Arduino inicializalasa
        if not self.serial_connection:
            print("FIGYELEM: Nincs csatlakoztatva Arduino!")
        
        aito_nyitva = False
        nyitva_ido = 0
        utolso_szemely = "Ismeretlen"
        utolso_bizonyossag = 0.0
        
        try:
            while True:
                ret, kep = kamera.read()
                if not ret:
                    continue
                    
                # Automatikus zaras idozitese
                if aito_nyitva and time.time() - nyitva_ido > self.beallitasok['nyitva_tartas']:
                    print("AUTOMATIKUS ZARAS")
                    aito_nyitva = False
                    self.zaras_vezerles("ZARAS")
                
                arc_kep, hatarolo, analizis = self.arc_eszleles_es_analizis(kep)
                
                if arc_kep is not None and not aito_nyitva:
                    nev, hasonlosag, adatbazis_tipus = self.arc_felismeres_cnn(arc_kep)
                    utolso_bizonyossag = hasonlosag
                    
                    if hasonlosag > self.beallitasok['kuszob']:
                        if adatbazis_tipus == "jovahagyott":
                            print(f"HOZZAFERES ENGEDELYEZVE: {nev} ({hasonlosag:.3f})")
                            aito_nyitva = True
                            nyitva_ido = time.time()
                            utolso_szemely = nev
                            self.zaras_vezerles("NYITAS")
                        elif adatbazis_tipus == "tiltott":
                            print(f"HOZZAFERES MEGTAGADVA: {nev} ({hasonlosag:.3f})")
                            utolso_szemely = f"TILTOTT: {nev}"
                
                # Felhasznaloi felulet rajzolasa
                self.biztonsagi_felirat_rajzolas(kep, hatarolo, aito_nyitva, 
                                               utolso_szemely, nyitva_ido, utolso_bizonyossag)
                cv2.imshow("FaceGate Biztonsagi Rendszer", kep)
                
                billentyu = cv2.waitKey(1) & 0xFF
                if billentyu == ord('q'):
                    break
                elif billentyu == ord('l') and aito_nyitva:
                    print("KEZI ZARAS")
                    aito_nyitva = False
                    self.zaras_vezerles("ZARAS")
                    
        finally:
            kamera.release()
            cv2.destroyAllWindows()

    def biztonsagi_felirat_rajzolas(self, kep, hatarolo, aito_nyitva, utolso_szemely, nyitva_ido, bizonyossag=0.0):
        """Biztonsagi felulet rajzolasa"""
        h, w = kep.shape[:2]
        
        # Atlatszo hatter az informaciokhoz
        overlay = kep.copy()
        cv2.rectangle(overlay, (0,0), (w,140), (0,0,0), -1)
        cv2.addWeighted(overlay, 0.7, kep, 0.3, 0, kep)
        
        # Allapot megjelenitese
        allapot_szin = (0, 255, 0) if aito_nyitva else (0, 0, 255)
        allapot_szoveg = "NYITVA" if aito_nyitva else "ZARVA"
        cv2.putText(kep, f"ALLAPOT: {allapot_szoveg}", (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, allapot_szin, 2)
        
        # CNN modell allapota
        cnn_allapot = "AKTIV" if self.modell_betoltve else "INAKTIV"
        cnn_szin = (0, 255, 0) if self.modell_betoltve else (0, 0, 255)
        cv2.putText(kep, f"CNN: {cnn_allapot}", (w-200, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, cnn_szin, 2)
        
        # Arduino allapot
        arduino_allapot = "CSATLAKOZVA" if self.serial_connection else "NINCS ARDUINO"
        arduino_szin = (0, 255, 0) if self.serial_connection else (0, 165, 255)
        cv2.putText(kep, f"ARDUINO: {arduino_allapot}", (w-300, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, arduino_szin, 1)
        
        # Utolso szemely
        cv2.putText(kep, f"UTOLSO FELISMERES: {utolso_szemely}", (20, 65), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Bizonyossag
        bizonyossag_szin = (0, 255, 0) if bizonyossag > self.beallitasok['kuszob'] else (0, 165, 255)
        cv2.putText(kep, f"BIZONYOSSAG: {bizonyossag:.3f}", (20, 95), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, bizonyossag_szin, 2)
        
        # Automatikus zaras visszaszamlalasa
        if aito_nyitva:
            hatralevo_ido = max(0, self.beallitasok['nyitva_tartas'] - (time.time() - nyitva_ido))
            cv2.putText(kep, f"AUTOMATIKUS ZARAS: {hatralevo_ido:.1f}s", (20, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Arc keretezes
        if hatarolo:
            x, y, w_rect, h_rect = hatarolo
            keret_szin = (0, 255, 0) if aito_nyitva else (0, 255, 255)
            cv2.rectangle(kep, (x, y), (x+w_rect, y+h_rect), keret_szin, 3)
            
            # Arc minoseg indikator
            minoseg_szoveg = "JO MINOSEG" 
            cv2.putText(kep, minoseg_szoveg, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    def adatbazis_mentese(self):
        """Mindket adatbazis mentese"""
        with open('data/jovahagyott_adatbazis.pkl', 'wb') as f:
            pickle.dump(self.jovahagyott_adatbazis, f)
        
        with open('data/tiltott_adatbazis.pkl', 'wb') as f:
            pickle.dump(self.tiltott_adatbazis, f)
        
        print("OK Adatbazisok elmentve")

    def statisztikak_megjelenitese(self):
        """Rendszer statisztikak megjelenitese"""
        print("\n" + "="*50)
        print("RENDSZER STATISZTIKAK")
        print("="*50)
        print(f"OK Jovahagyott szemelyek: {len(self.jovahagyott_adatbazis)}")
        print(f"OK Tiltott szemelyek: {len(self.tiltott_adatbazis)}")
        
        osszes_minta = sum(adatok['mintak_szama'] for adatok in self.jovahagyott_adatbazis.values())
        osszes_minta += sum(adatok['mintak_szama'] for adatok in self.tiltott_adatbazis.values())
        
        print(f"OK Osszes minta: {osszes_minta}")
        print(f"OK Biztonsagi kuszob: {self.beallitasok['kuszob']}")
        print(f"OK CNN modell allapot: {'BETOLTVE' if self.modell_betoltve else 'NINCS BETOLTVE'}")
        print(f"OK Arduino csatlakozas: {'AKTIV' if self.serial_connection else 'INAKTIV'}")
        print(f"OK Osztalyok szama: {len(self.osztaly_nev_kodolas)}")

    def szemelyek_listazasa(self):
        """Regisztralt szemelyek listazasa"""
        print("\n" + "="*50)
        print("REGISZTRALT SZEMELYEK")
        print("="*50)
        
        if not self.jovahagyott_adatbazis and not self.tiltott_adatbazis:
            print("INFO Nincsenek regisztralt szemelyek")
            return
        
        if self.jovahagyott_adatbazis:
            print("\nJOVAHAGYOTT SZEMELYEK:")
            for nev, adatok in self.jovahagyott_adatbazis.items():
                reg_datum = datetime.fromtimestamp(adatok['regisztracio_ido']).strftime('%Y-%m-%d %H:%M')
                print(f"  {nev} - {adatok['mintak_szama']} minta - {reg_datum}")
        
        if self.tiltott_adatbazis:
            print("\nTILTOTT SZEMELYEK:")
            for nev, adatok in self.tiltott_adatbazis.items():
                reg_datum = datetime.fromtimestamp(adatok['regisztracio_ido']).strftime('%Y-%m-%d %H:%M')
                print(f"  {nev} - {adatok['mintak_szama']} minta - {reg_datum}")

    def cnn_tanulasi_statusz(self):
        """CNN tanulasi statusz megjelenitese"""
        print("\n" + "="*50)
        print("CNN TANULASI STATUSZ")
        print("="*50)
        
        print(f"OK Modell allapot: {'BETOLTVE' if self.modell_betoltve else 'NINCS BETOLTVE'}")
        print(f"OK Osztalyok szama: {len(self.osztaly_nev_kodolas)}")
        print(f"OK Jovahagyott szemelyek: {len(self.jovahagyott_adatbazis)}")
        print(f"OK Tiltott szemelyek: {len(self.tiltott_adatbazis)}")
        
        osszes_minta = sum(adatok['mintak_szama'] for adatok in self.jovahagyott_adatbazis.values())
        osszes_minta += sum(adatok['mintak_szama'] for adatok in self.tiltott_adatbazis.values())
        print(f"OK Osszes tanito minta: {osszes_minta}")
        
        if osszes_minta < 50:
            print("FIGYELEM: Keves tanito adat! Ajanlott minimum 50 minta.")
        else:
            print("OK Megfelelo mennyisegu tanito adat.")

    def beallitasok_menu(self):
        """Beallitasok menu"""
        while True:
            print("\n" + "="*50)
            print("BEALLITASOK")
            print("="*50)
            print("1. Kamera port beallitasa")
            print("2. Arduino port beallitasa")
            print("3. Rendszer beallitasok")
            print("4. Statusz megjelenitese")
            print("5. About")
            print("6. Vissza a fomenube")
            
            valasztas = input("\nValasztas (1-6): ").strip()
            
            if valasztas == '1':
                port = self.port_valasztas("kamera")
                if port:
                    self.beallitasok['kamera_index'] = int(port)
                    print(f"OK Kamera beallitva: {port}")
            elif valasztas == '2':
                self.arduino_csatlakozas()
            elif valasztas == '3':
                self.rendszer_beallitasok()
            elif valasztas == '4':
                self.statisztikak_megjelenitese()
            elif valasztas == '5':
                self.about_info()
            elif valasztas == '6':
                break
            else:
                print("ERROR Ervenytelen valasztas!")

    def rendszer_beallitasok(self):
        """Rendszer beallitasok modositasa"""
        while True:
            print("\n" + "="*50)
            print("RENDSZER BEALLITASOK")
            print("="*50)
            print(f"1. Biztonsagi kuszob: {self.beallitasok['kuszob']}")
            print(f"2. Nyitva tartas ideje: {self.beallitasok['nyitva_tartas']} masodperc")
            print(f"3. Maximum mintak: {self.beallitasok['max_mintak']}")
            print(f"4. Tanitasi epochok: {self.beallitasok['tanitas_epochs']}")
            print(f"5. Batch meret: {self.beallitasok['batch_size']}")
            print(f"6. Automatikus rogzites kesleltetes: {self.beallitasok['auto_capture_delay']} masodperc")
            print("7. Vissza")
            
            valasztas = input("\nValasztas (1-7): ").strip()
            
            if valasztas == '1':
                try:
                    uj_kuszob = float(input(f"Uj kuszobertek ({self.beallitasok['kuszob']}): "))
                    if 0.1 <= uj_kuszob <= 1.0:
                        self.beallitasok['kuszob'] = uj_kuszob
                        print(f"OK Kuszobertek beallitva: {uj_kuszob}")
                    else:
                        print("ERROR A kuszobertek 0.1 es 1.0 kozott legyen!")
                except ValueError:
                    print("ERROR Ervenytelen ertek!")
                    
            elif valasztas == '2':
                try:
                    uj_ido = int(input(f"Uj nyitva tartasi ido ({self.beallitasok['nyitva_tartas']}): "))
                    if uj_ido > 0:
                        self.beallitasok['nyitva_tartas'] = uj_ido
                        print(f"OK Nyitva tartasi ido beallitva: {uj_ido} masodperc")
                    else:
                        print("ERROR Az ido pozitiv legyen!")
                except ValueError:
                    print("ERROR Ervenytelen ertek!")
                    
            elif valasztas == '3':
                try:
                    uj_max = int(input(f"Uj maximum mintak ({self.beallitasok['max_mintak']}): "))
                    if uj_max >= 15:
                        self.beallitasok['max_mintak'] = uj_max
                        print(f"OK Maximum mintak beallitva: {uj_max}")
                    else:
                        print("ERROR Legalabb 15 minta szukseges!")
                except ValueError:
                    print("ERROR Ervenytelen ertek!")
                    
            elif valasztas == '4':
                try:
                    uj_epochs = int(input(f"Uj epochok ({self.beallitasok['tanitas_epochs']}): "))
                    if uj_epochs > 0:
                        self.beallitasok['tanitas_epochs'] = uj_epochs
                        print(f"OK Epochok beallitva: {uj_epochs}")
                    else:
                        print("ERROR Az epochok szam pozitiv legyen!")
                except ValueError:
                    print("ERROR Ervenytelen ertek!")
                    
            elif valasztas == '5':
                try:
                    uj_batch = int(input(f"Uj batch meret ({self.beallitasok['batch_size']}): "))
                    if uj_batch > 0:
                        self.beallitasok['batch_size'] = uj_batch
                        print(f"OK Batch meret beallitva: {uj_batch}")
                    else:
                        print("ERROR A batch meret pozitiv legyen!")
                except ValueError:
                    print("ERROR Ervenytelen ertek!")
                    
            elif valasztas == '6':
                try:
                    uj_delay = float(input(f"Uj automatikus rogzites kesleltetes ({self.beallitasok['auto_capture_delay']}): "))
                    if uj_delay > 0:
                        self.beallitasok['auto_capture_delay'] = uj_delay
                        print(f"OK Automatikus rogzites kesleltetes beallitva: {uj_delay} masodperc")
                    else:
                        print("ERROR A kesleltetes pozitiv legyen!")
                except ValueError:
                    print("ERROR Ervenytelen ertek!")
                    
            elif valasztas == '7':
                break
            else:
                print("ERROR Ervenytelen valasztas!")

    def about_info(self):
        """About informacio megjelenitese"""
        print("\n" + "="*60)
        print("FACE GATE - CNN ARC FELISMERO RENDSZER")
        print("="*60)
        
        for key, value in self.rendszer_info.items():
            print(f"{key.upper():<15}: {value}")
        
        print("\n" + "-"*60)
        print("A rendszer mely neuralis halozatot hasznal arc felismeresre.")
        print("Kepes tanulni uj arcokat es valos idoben vegezni az azonositast.")
        print("-"*60)

    def lista_menu(self):
        """Lista menu"""
        while True:
            print("\n" + "="*50)
            print("LISTA MENU")
            print("="*50)
            print("1. Engedelyezett hozzaadasa")
            print("2. Tiltott hozzaadasa")
            print("3. CNN tanulasi statusz")
            print("4. Szemelyek listazasa")
            print("5. CNN modell ujratanitasa")
            print("6. Vissza a fomenube")
            
            valasztas = input("\nValasztas (1-6): ").strip()
            
            if valasztas == '1':
                nev = input("Szemely neve: ").strip()
                if nev:
                    self.arc_regisztralasa_auto(nev, "jovahagyott")
                else:
                    print("ERROR Ervenytelen nev!")
            elif valasztas == '2':
                nev = input("Szemely neve: ").strip()
                if nev:
                    self.arc_regisztralasa_auto(nev, "tiltott")
                else:
                    print("ERROR Ervenytelen nev!")
            elif valasztas == '3':
                self.cnn_tanulasi_statusz()
            elif valasztas == '4':
                self.szemelyek_listazasa()
            elif valasztas == '5':
                self.cnn_tanitas()
            elif valasztas == '6':
                break
            else:
                print("ERROR Ervenytelen valasztas!")

    def fo_menu(self):
        """Fomenu megjelenitese es kezelese"""
        # Rendszer inditasi informaciok typewriter effect-tel
        print("\n" + "*" * 60)
        self.typewriter_effect("FACE GATE - CNN ARC FELISMERO RENDSZER", 0.05)
        print("*" * 60)
        time.sleep(0.5)
        
        for key, value in self.rendszer_info.items():
            self.typewriter_effect(f"{key.upper():<15}: {value}", 0.02)
            time.sleep(0.1)
        
        print("\n" + ">" * 60)
        self.typewriter_effect("A rendszer elindult. Kerjuk, allitsd be a kamerat es az Arduino-t!", 0.03)
        print(">" * 60 + "\n")
        time.sleep(1)
        
        while True:
            print("\n" + "="*60)
            print("FOMENU - FACE GATE 1.1.0")
            print("="*60)
            print("1. AKTIVALAS (Biztonsagi rendszer inditasa)")
            print("2. LISTA (Arcok kezelese)")
            print("3. BEALLITASOK (Rendszer konfiguralasa)")
            print("4. KILEPES")
            
            valasztas = input("\nValasztas (1-4): ").strip()
            
            if valasztas == '1':
                self.biztonsagi_rendszer_inditasa()
            elif valasztas == '2':
                self.lista_menu()
            elif valasztas == '3':
                self.beallitasok_menu()
            elif valasztas == '4':
                if self.serial_connection:
                    self.serial_connection.close()
                print("\n" + "="*50)
                print("RENDSZER LEALLITVA")
                print("Koszönjük, hogy hasznalta a FaceGate rendszert!")
                print("="*50)
                break
            else:
                print("ERROR Ervenytelen valasztas!")

if __name__ == "__main__":
    # TensorFlow warning-ok elrejtese
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    # Rendszer inditasa
    rendszer = BiztonsagiArcRendszer()
    rendszer.fo_menu()
