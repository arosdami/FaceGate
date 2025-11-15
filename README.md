# FaceGate 1.0.8

## Áttekintés

A **FaceGate** egy olyan biztonsági rendszer, amely arcfelismerést
használ ajtók automatikus nyitására és zárására. A rendszer kameraképből
azonosítja a személyeket, és ha felismeri őket, engedélyezi a belépést.
Ismeretlen személy esetén megtagadja a hozzáférést.

## Főbb jellemzők

-   **Többplatformos támogatás** -- Windows, Linux és macOS
    kompatibilitás\
-   **Titkosított biometrikus tárolás** -- XOR-alapú titkosított arci
    jellemzőpont tárolás\
-   **Kettős felismerési módszer** -- CNN beágyazások + LBPH algoritmus\
-   **Valós idejű feldolgozás** -- Arcfelismerés élő kamera képen\
-   **Arduino integráció** -- Fizikai ajtóvezérlés támogatása\
-   **Többnyelvű felület** -- Magyar és angol nyelv\
-   **Biztonságos adatbázis** -- Titkosított személyadatbázis,
    automatikus mentéssel

## Rendszerkövetelmények

### Hardver követelmények

-   USB vagy beépített kamera\
-   Legalább 4 GB RAM\
-   Minimum 2 GB szabad tárhely\
-   **Arduino Uno** (opcionális, ajtóvezérléshez)

### Szoftver követelmények

-   Python **3.12.4**
-   OpenCV **4.5+**
-   TensorFlow **2.8+**
-   NumPy
-   PySerial (Arduino kommunikáció)

## Telepítés

### 1. Python és függőségek telepítése

Kérjük, telepítse a Python 3.12.4 verziót, majd a szükséges csomagokat:

``` bash
pip install -r requirements.txt
```

A `requirements.txt` tartalma:

    opencv-python==4.8.1.78
    tensorflow==2.13.0
    numpy==1.24.3
    pyserial==3.5

### 2. Arduino beállítása

Az **Arduino Uno** mikrovezérlőre töltse fel a `C_UNLOCK.cpp` fájlt. Ez
felel az ajtó vezérléséért a rendszer parancsai alapján.

### 3. A rendszer indítása

``` bash
python main.py
```

## Használat

### Főmenü opciók

1.  **Új arc regisztrálása (Titkosított)**
2.  **Biztonsági rendszer indítása**
3.  **Rendszer beállítások**
4.  **Rendszerinformációk**
5.  **Adatbázis kezelés**
6.  **Nyelv váltás**
7.  **Kilépés**

## Fájlstruktúra

``` plaintext
FaceGate/
├── main.py
├── requirements.txt
├── C_UNLOCK.cpp
├── data/
│   ├── secure_facegate_database.pkl
│   ├── encryption_key.key
│   └── secure_faces/
├── models/
│   └── secure_facegate_lbph_model.xml
├── backups/
├── logs/
└── README.md
```

## Licenc

Ez a projekt a Tokaj-Hegyalja Egyetem Robotika, MI & NN kurzusán készült
oktatási célra.

Fejlesztő: Damjan Aros – THE PTI
Projekt: Egyetemi projekt – FaceGate
Egyetem: Tokaj-Hegyalja Egyetem
Kurzus: Robotika, Mesterséges Intelligencia & Neurális Hálózatok
Témavezetők: Attila Perlaki, Dávid Gegény
Félév: 2025 Ősz
