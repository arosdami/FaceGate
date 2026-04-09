# FaceGate 1.1.5

![Videó_Készítés_és_Megosztás](https://github.com/user-attachments/assets/58d493a1-0473-4f03-9195-e65e4178b0bb)


![1118(1)](https://github.com/user-attachments/assets/9face1b4-724f-4a78-9ab0-20580808807c)

---

## English Documentation

## Overview
FaceGate is a professional-grade security solution that implements comprehensive facial recognition access control using state-of-the-art artificial intelligence techniques. The system is not merely a simple recognition application, but a complex, self-learning security infrastructure that seamlessly integrates biometric identification, real-time image processing, and physical access control.

The fundamental operating principle of FaceGate is a multi-layered security architecture that ensures reliable personal identification through the application of modern convolutional neural networks (CNNs). The system's specially designed neural architecture enables not only static facial recognition but also continuous learning and adaptation, guaranteeing long-term accuracy and reliability.

The system's outstanding technological feature is real-time adaptive decision-making, which far exceeds the limitations of traditional facial recognition systems by dynamically adapting to changing environmental conditions (lighting conditions, camera angles, facial expressions). This capability is based on the synergy of specially trained neural networks and complex preprocessing algorithms.

Physical security integration is provided by an intelligent Arduino-based actuator system that transforms software decisions into real-time physical actions. This solution enables not only automatic door control but also continuous monitoring of system status and detailed logging of security events.

The system's unique design element is proactive security behavior, which results in immediate detection of unknown faces and instant denial of access. This functionality implements a preventive security approach compared to traditional reactive systems, significantly increasing the security level of protected areas.

<img width="1024" height="1024" alt="Gemini_Generated_Image_d3wjeyd3wjeyd3wj" src="https://github.com/user-attachments/assets/c3cbeea2-7fb2-4de9-9ee3-ba1b49c57984" />

## Key Features
- **CNN-based facial recognition** - Custom neural network for accurate recognition
- **Automatic learning** - System capable of continuously learning new faces
- **Real-time processing** - Face recognition on live camera feed
- **Arduino integration** - Physical door control support
- **Intelligent security** - Instant detection of unknown faces
- **Timed opening** - 10-second opening for known faces
- **Menu system** - Full graphical user interface

## System Requirements

### Hardware Requirements
- USB or built-in camera
- Minimum 8 GB RAM (16 GB recommended for CNN training)
- At least 5 GB free storage space
- Arduino Uno (optional, for door control)
- GPU (optional but recommended for faster CNN processing)

### Software Requirements
- Python 3.8+
- OpenCV 4.5+
- TensorFlow 2.8+
- NumPy
- PySerial (Arduino communication)
- scikit-learn

## Installation

### 1. Python and Dependencies Installation
Install the required packages:

    pip install -r requirements.txt

**requirements.txt content:**

    torch>=1.9.0
    torchvision>=0.10.0
    opencv-python>=4.5.0
    numpy>=1.21.0
    scikit-learn>=1.0.0
    scikit-image>=0.19.0
    Pillow>=9.0.0
    pyserial>=3.5
    facenet-pytorch>=2.5.0

### 2. Arduino Setup
Upload the C_UNLOCK.cpp file to the Arduino Uno microcontroller. This is responsible for door control.

    #include <Servo.h>
    Servo lockServo;
    const int SERVO_PIN = 3;
    
    void setup() {
      lockServo.attach(SERVO_PIN);
      Serial.begin(9600);
      lockServo.write(90);
      Serial.println("UNO READY - D3 SERVO");
    }
    
    void loop() {
      if (Serial.available() > 0) {
        String cmd = Serial.readStringUntil('\n');
        cmd.trim();
        
        if (cmd == "UNLOCK") {
          lockServo.write(0);
          Serial.println("NYITVA");
        }
        else if (cmd == "LOCK") {
          lockServo.write(90);
          Serial.println("ZARVA");
        }
      }
    }

### 3. System Startup

    python main.py

## Usage

### Main Menu Options
- **Start Face Recognition** - Activate security system
- **Learn New Face** - Add new person to the system
- **Known Faces** - Manage registered faces
- **Settings** - Configure system
- **Exit** - Close program

## Menu Navigation
- **W** - Up
- **S** - Down
- **ENTER** - Select
- **ESC** - Back/Exit
- **M** - Main Menu

<img width="627" height="480" alt="image" src="https://github.com/user-attachments/assets/04597f68-2ea3-4f51-bbc0-d13550dafa67" />

## Face Registration
- **Automatic pattern collection** - 15 images captured automatically
- **Real-time quality check** - Face position and quality verification
- **Neural network training** - Automatic model update after new faces
  
## Security Mode
The system:
- Monitors camera in real-time
- Automatically recognizes registered faces
- Controls Arduino door lock
- Performs automatic locking after set time
- Creates detailed logs of recognitions
- Handles and recognizes multiple faces simultaneously

## System Settings
- **Security threshold**: 0.1–1.0 (default: 0.95)
- **Opening duration**: 1–60 seconds (default: 8)
- **Maximum samples**: 15–200 (default: 50)
- **Training epochs**: 10–100 (default: 30)
- **Batch size**: 8–64 (default: 16)
- **Automatic capture delay**: 0.1–2.0 seconds (default: 0.5)

## File Structure

    FaceGate/
    ├── main.py
    ├── config.py
    ├── neural_face_recognizer.py
    ├── arduino_controller.py
    ├── menu_system.py
    ├── requirements.txt
    ├── face_model.pth
    ├── known_faces.pkl
    └── system_config.json

## CNN Architecture
The system uses 4 convolutional layers:

1. **Convolutional block** — 32 filters, BatchNormalization, MaxPooling, Dropout
2. **Convolutional block** — 64 filters, BatchNormalization, MaxPooling, Dropout
3. **Convolutional block** — 128 filters, BatchNormalization, MaxPooling, Dropout

**Fully connected layers**: 512 and 256 neurons + Dropout  
**Output layer**: Softmax activation

## Operating Principle
- **Face detection** - Haar cascade algorithm
- **Feature extraction** - CNN neural network for facial feature extraction
- **Similarity calculation** - Cosine similarity with known faces
- **Decision** - Threshold-based identification
- **Control** - Arduino command sending

## License
This project was created for educational purposes at the Tokaj-Hegyalja University Robotics, AI & NN course.

**Developer**: Damjan Aros – THE PTI  
**Project**: University Project – FaceGate  
**University**: Tokaj-Hegyalja University  
**Course**: Robotics, Artificial Intelligence & Neural Networks  
**Supervisors**: Attila Perlaki, Dávid Gégény  
**Semester**: 2025 Fall

<img width="639" height="510" alt="image" src="https://github.com/user-attachments/assets/07b02024-777f-43b4-9691-b0fa23740fcb" />

---

## Magyar Dokumentáció

## Áttekintés
A FaceGate egy professzionális szintű biztonsági megoldás, amely a mesterséges intelligencia legkorszerűbb technikáit alkalmazva valósít meg teljes körű arcfelismeréses hozzáférés-vezérlést. A rendszer nem csupán egy egyszerű felismerő alkalmazás, hanem egy komplex, önállóan tanulni képes biztonsági infrastruktúra, amely a biometrikus azonosítás, valós idejű képfeldolgozás és fizikai access control zökkenőmentes integrációját valósítja meg.

A FaceGate alapvető működési elve a többrétegű biztonsági architektúra, amely a legmodernebb konvolúciós neurális hálózatok (CNN) alkalmazásán keresztül biztosítja a megbízható személyazonosítást. A rendszer speciálisan tervezett neurális architektúrája lehetővé teszi nemcsak a statikus arcfelismerést, hanem a folyamatos tanulást és adaptációt is, ezzel garantálva a hosszú távú pontosságot és megbízhatóságot.

A rendszer kiemelkedő technológiai jellemzője a valós idejű adaptív döntéshozatal, amely a hagyományos arcfelismerő rendszerek korlátait messze meghaladva, képes dinamikusan alkalmazkodni a változó környezeti feltételekhez (világítási viszonyok, kameraszögek, arckifejezések). Ez a képesség a speciálisan kiképzett neurális háló és a komplex előfeldolgozó algoritmusok szinergiáján alapul.

A fizikai biztonsági integrációt egy intelligens Arduino-alapú aktuátorrendszer biztosítja, amely a szoftveres döntéseket valós idejű fizikai akciókká alakítja. Ez a megoldás lehetővé teszi nemcsak az ajtók automatikus vezérlését, hanem a rendszer állapotának folyamatos monitorozását és a biztonsági események részletes naplózását is.

A rendszer egyedi tervezési eleme a proaktív biztonsági viselkedés, amely az ismeretlen arcok azonnali észlelését és a hozzáférés azonnali megtagadását eredményezi. Ez a funkcionalitás a hagyományos reakcióalapú rendszerekkel szemben preventív biztonsági megközelítést valósít meg, jelentősen növelve a védett területek biztonsági szintjét.

## Főbb jellemzők
- CNN alapú arcfelismerés - Saját neurális háló pontos felismeréshez
- Automatikus tanulás - A rendszer képes új arcok tanulására folyamatosan
- Valós idejű feldolgozás - Arc felismerés élő kamera képen
- Arduino integráció — Fizikai ajtóvezérlés támogatása
- Intelligens biztonság - Ismeretlen arc azonnali észlelése
- Időzített nyitás - 10 másodperc nyitva tartás ismert arcoknak
- Menu rendszer - Teljes grafikus kezelőfelület

## Rendszerkövetelmények

### Hardver követelmények
- USB vagy beépített kamera
- Legalább 8 GB RAM (ajánlott 16 GB CNN tanításhoz)
- Minimum 5 GB szabad tárhely
- Arduino Uno (opcionális, ajtóvezérléshez)
- GPU (opcionális, de ajánlott gyorsabb CNN feldolgozáshoz)

### Szoftver követelmények
- Python 3.8+
- OpenCV 4.5+
- TensorFlow 2.8+
- NumPy
- PySerial (Arduino kommunikáció)
- scikit-learn

## Telepítés

### 1. Python és függőségek telepítése
A szükséges csomagok telepítése:

    pip install -r requirements.txt

**requirements.txt tartalma:**

    torch>=1.9.0
    torchvision>=0.10.0
    opencv-python>=4.5.0
    numpy>=1.21.0
    scikit-learn>=1.0.0
    scikit-image>=0.19.0
    Pillow>=9.0.0
    pyserial>=3.5
    facenet-pytorch>=2.5.0

### 2. Arduino beállítása
Az Arduino Uno mikrovezérlőre töltse fel a C_UNLOCK.cpp fájlt. Ez felel az ajtó vezérléséért.

    #include <Servo.h>
    Servo lockServo;
    const int SERVO_PIN = 3;
    
    void setup() {
      lockServo.attach(SERVO_PIN);
      Serial.begin(9600);
      lockServo.write(90);
      Serial.println("UNO READY - D3 SERVO");
    }
    
    void loop() {
      if (Serial.available() > 0) {
        String cmd = Serial.readStringUntil('\n');
        cmd.trim();
        
        if (cmd == "UNLOCK") {
          lockServo.write(0);
          Serial.println("NYITVA");
        }
        else if (cmd == "LOCK") {
          lockServo.write(90);
          Serial.println("ZARVA");
        }
      }
    }

### 3. A rendszer indítása

    python main.py

## Használat

### Főmenü opciók
- Arc Felismeres Inditasa - Biztonsági rendszer aktiválása
- Uj Arc Tanulasa - Új személy hozzáadása a rendszerhez
- Ismert Arcok - Regisztrált arcok kezelése
- Beallitasok - Rendszer konfigurálása
- Kilepes - Program bezárása

## Navigáció a menüben
- W - Fel
- S - Le
- ENTER - Kiválasztás
- ESC - Vissza/Kilépés
- M - Főmenü

## Arc regisztrálás
- Automata minta gyűjtés - 15 kép automatikus rögzítése
- Valós idejű minőségellenőrzés - Arc pozíció és minőség ellenőrzése
- Neurális háló tanítás - Automatikus modell frissítés új arcok után
  
## Biztonsági mód
A rendszer:
- Valós időben figyeli a kamerát
- Automatikusan felismeri a regisztrált arcokat
- Vezérli az Arduino ajtózárát
- Automatikus zárást végez beállított idő után
- Részletes naplózást készít a felismerésekről
- Több arc egyidejű kezelése és felismerése

## Rendszer beállítások
- Biztonsági küszöb: 0.1–1.0 (alapértelmezett: 0.95)
- Nyitva tartás ideje: 1–60 mp (alapértelmezett: 8)
- Maximum minták: 15–200 (alapértelmezett: 50)
- Tanítási epochok: 10–100 (alapértelmezett: 30)
- Batch méret: 8–64 (alapértelmezett: 16)
- Automatikus rögzítés késleltetés: 0.1–2.0 s (alapértelmezett: 0.5)

## Fájlstruktúra

    FaceGate/
    ├── main.py
    ├── config.py
    ├── neural_face_recognizer.py
    ├── arduino_controller.py
    ├── menu_system.py
    ├── requirements.txt
    ├── face_model.pth
    ├── known_faces.pkl
    └── system_config.json

## CNN Architektúra
A rendszer 4 konvolúciós réteget használ:

1. Konvolúciós blokk — 32 szűrő, BatchNormalization, MaxPooling, Dropout
2. Konvolúciós blokk — 64 szűrő, BatchNormalization, MaxPooling, Dropout
3. Konvolúciós blokk — 128 szűrő, BatchNormalization, MaxPooling, Dropout

Teljesen összekapcsolt rétegek: 512 és 256 neuron + Dropout  
Kimeneti réteg: Softmax aktiváció

##Működési elv
- Arc detektálás - Haar cascade algoritmus
- Feature extraction - CNN neurális háló arc jellemzők kinyerésére
- Hasonlóság számítás - Koszinusz hasonlóság ismert arcokkal
- Döntés - Küszöbérték alapú azonosítás
- Vezérlés - Arduino parancsok küldése

## Licenc
Ez a projekt a Tokaj-Hegyalja Egyetem Robotika, MI & NN kurzusán készült oktatási célra.

Fejlesztő: Damjan Aros – THE PTI  
Projekt: Egyetemi projekt – FaceGate  
Egyetem: Tokaj-Hegyalja Egyetem  
Kurzus: Robotika, Mesterséges Intelligencia & Neurális Hálózatok  
Témavezetők: Attila Perlaki, Dávid Gégény  
Félév: 2025 Ősz

