# FaceGate 1.1.0

## Áttekintés
A FaceGate egy olyan biztonsági rendszer, amely arcfelismerést használ ajtók automatikus nyitására és zárására. A rendszer kameraképből azonosítja a személyeket, és ha felismeri őket, engedélyezi a belépést. Ismeretlen személy esetén megtagadja a hozzáférést.

## Főbb jellemzők
- CNN alapú arcfelismerés — Mély neurális hálózat pontos felismeréshez
- Automatikus tanulás — A rendszer képes új arcok tanulására és adaptálására
- Valós idejű feldolgozás — Arcfelismerés élő kamera képen
- Arduino integráció — Fizikai ajtóvezérlés támogatása
- Minőségellenőrzés — Automatikus arcminőség értékelés adatgyűjtéskor
- Kettős adatbázis — Jóváhagyott és tiltott személyek külön kezelése
- Automatikus adatgyűjtés — Intelligens minta rögzítés kézi beavatkozás nélkül

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

    opencv-python==4.8.1.78
    tensorflow==2.13.0
    numpy==1.24.3
    pyserial==3.5
    scikit-learn==1.3.0

### 2. Arduino beállítása
Az Arduino Uno mikrovezérlőre töltse fel a C_UNLOCK.cpp fájlt. Ez felel az ajtó vezérléséért.

### 3. A rendszer indítása

    python main.py

## Használat

### Főmenü opciók
- Aktiválás — Biztonsági rendszer indítása
- Lista — Arcok kezelése
  - Engedélyezett hozzáadása
  - Tiltott hozzáadása
  - CNN tanulási státusz
  - Személyek listázása
  - CNN modell újratanítása
- Beállítások — Rendszer konfigurálása
  - Kamera port beállítása
  - Arduino port beállítása
  - Rendszer beállítások
- Státusz megjelenítése
- About információ
- Kilépés

## Arc regisztrálás
- Automatikus adatgyűjtés jó minőségű arc képekkel
- Minimum 15 minta szükséges
- Valós idejű minőségellenőrzés (fényesség, kontraszt, szemek)
- Automatikus CNN modell frissítés új arcok után

## Biztonsági mód
A rendszer:
- Valós időben figyeli a kamerát
- Automatikusan felismeri a regisztrált arcokat
- Vezérli az Arduino ajtózárát
- Automatikus zárást végez beállított idő után
- Részletes naplózást készít a felismerésekről

## Rendszer beállítások
- Biztonsági küszöb: 0.1–1.0 (alapértelmezett: 0.85)
- Nyitva tartás ideje: 1–60 mp (alapértelmezett: 8)
- Maximum minták: 15–200 (alapértelmezett: 50)
- Tanítási epochok: 10–100 (alapértelmezett: 30)
- Batch méret: 8–64 (alapértelmezett: 16)
- Automatikus rögzítés késleltetés: 0.1–2.0 s (alapértelmezett: 0.5)

## Fájlstruktúra

    FaceGate/
    ├── main.py
    ├── requirements.txt
    ├── Dokumentacio
    ├── Arduino/
    ├── data/
    │   ├── jovahagyott/
    │   ├── tiltott/
    │   ├── jovahagyott_adatbazis.pkl
    │   ├── osztaly_kodolas.pkl
    │   └── tiltott_adatbazis.pkl
    ├── models/
    │   └── arc_felismero_model.h5
    ├── training_data
    ├── LICENSE
    └── README.md

## CNN Architektúra
A rendszer 4 konvolúciós réteget használ:

1. Konvolúciós blokk — 32 szűrő, BatchNormalization, MaxPooling, Dropout
2. Konvolúciós blokk — 64 szűrő, BatchNormalization, MaxPooling, Dropout
3. Konvolúciós blokk — 128 szűrő, BatchNormalization, MaxPooling, Dropout
4. Konvolúciós blokk — 256 szűrő, BatchNormalization, MaxPooling, Dropout

Teljesen összekapcsolt rétegek: 512 és 256 neuron + Dropout  
Kimeneti réteg: Softmax aktiváció

## Licenc
Ez a projekt a Tokaj-Hegyalja Egyetem Robotika, MI & NN kurzusán készült oktatási célra.

Fejlesztő: Damjan Aros – THE PTI  
Projekt: Egyetemi projekt – FaceGate  
Egyetem: Tokaj-Hegyalja Egyetem  
Kurzus: Robotika, Mesterséges Intelligencia & Neurális Hálózatok  
Témavezetők: Attila Perlaki, Dávid Gégény  
Félév: 2025 Ősz
