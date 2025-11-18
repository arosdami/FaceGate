# FaceGate 1.1.5


![1118(1)](https://github.com/user-attachments/assets/9face1b4-724f-4a78-9ab0-20580808807c)

## Áttekintés
A FaceGate egy professzionális szintű biztonsági megoldás, amely a mesterséges intelligencia legkorszerűbb technikáit alkalmazva valósít meg teljes körű arcfelismeréses hozzáférés-vezérlést. A rendszer nem csupán egy egyszerű felismerő alkalmazás, hanem egy komplex, önállóan tanulni képes biztonsági infrastruktúra, amely a biometrikus azonosítás, valós idejű képfeldolgozás és fizikai access control zökkenőmentes integrációját valósítja meg.

A FaceGate alapvető működési elve a többrétegű biztonsági architektúra, amely a legmodernebb konvolúciós neurális hálózatok (CNN) alkalmazásán keresztül biztosítja a megbízható személyazonosítást. A rendszer speciálisan tervezett neurális architektúrája lehetővé teszi nemcsak a statikus arcfelismerést, hanem a folyamatos tanulást és adaptációt is, ezzel garantálva a hosszú távú pontosságot és megbízhatóságot.

A rendszer kiemelkedő technológiai jellemzője a valós idejű adaptív döntéshozatal, amely a hagyományos arcfelismerő rendszerek korlátait messze meghaladva, képes dinamikusan alkalmazkodni a változó környezeti feltételekhez (világítási viszonyok, kameraszögek, arckifejezések). Ez a képesség a speciálisan kiképzett neurális háló és a komplex előfeldolgozó algoritmusok szinergiáján alapul.

A fizikai biztonsági integrációt egy intelligens Arduino-alapú aktuatórrendszer biztosítja, amely a szoftveres döntéseket valós idejű fizikai akciókká alakítja. Ez a megoldás lehetővé teszi nemcsak az ajtók automatikus vezérlését, hanem a rendszer állapotának folyamatos monitorozását és a biztonsági események részletes naplózását is.

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

<img width="627" height="480" alt="image" src="https://github.com/user-attachments/assets/04597f68-2ea3-4f51-bbc0-d13550dafa67" />


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

<img width="639" height="510" alt="image" src="https://github.com/user-attachments/assets/07b02024-777f-43b4-9691-b0fa23740fcb" />

