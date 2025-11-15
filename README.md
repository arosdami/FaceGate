FaceGate Security System
========================

Áttekintés
----------

A FaceGate egy olyan program, ami arcfelismeréssel működő ajtó nyitó rendszer. A kamera felismeri az emberek arcát, és ha ismeri őket, kinyitja az ajtót. Ha nem ismeri, nem enged be.

Főbb jellemzők
--------------

*   **Többplatformos támogatás**: Windows, Linux és macOS kompatibilitás
    
*   **Titkosított biometrikus tárolás**: XOR titkosított arci jellemzőpont tárolás
    
*   **Kettős felismerési módszer**: CNN beágyazások + LBPH hagyományos felismerés
    
*   **Valós idejű feldolgozás**: Élő kamera kép arcfelismeréssel
    
*   **Arduino integráció**: Fizikai hozzáférés-vezérlő rendszer támogatás
    
*   **Többnyelvű felület**: Magyar és angol nyelv támogatása
    
*   **Biztonságos adatbázis**: Titkosított személy adatbázis biztonsági mentéssel
    

Rendszerkövetelmények
---------------------

### Hardver követelmények

*   Kamera (USB webkamera vagy beépített)
    
*   Minimum 4GB RAM
    
*   2GB szabad lemezterület
    
*   Arduino Uno (opcionális, ajtó vezérléshez)
    

### Szoftver követelmények

*   Python 3.12.4
    
*   OpenCV 4.5+
    
*   TensorFlow 2.8+
    
*   NumPy
    
*   PySerial (Arduino kommunikációhoz)
    

Telepítés
---------

### 1\. Python és függőségek telepítése

Először telepítsd a Python 3.12.4 verziót, majd a szükséges könyvtárakat:

*** pip install -r requirements.txt ***

A requirements.txt tartalma:

opencv-python==4.8.1.78
tensorflow==2.13.0
numpy==1.24.3
pyserial==3.5

### 2\. Arduino beállítása

Az Arduino Uno mikrovezérlőhöz töltsd fel a C\_UNLOCK.cpp fájlt. Ez a kód kezeli az ajtó nyitását/zárását a FaceGate rendszer parancsai alapján.

Az Arduino kód főbb funkciói:

*   Soros kommunikáció a számítógéppel
    
*   AJtó nyitás/zárás vezérlése
    
*   LED indikációk
    
*   Biztonsági időzítések
    

### 3\. Rendszer indítása

A telepítés után indítsd el a rendszert:

***python main.py***

### Használat

### Főmenü opciók

1.  **Új arc regisztrálása (Titkosított)**
    
    *   Új felhasználók regisztrálása titkosított arci adattárolással
        
    *   30 db jó minőségű arcminta gyűjtése
        
    *   Titkosított jellemzőpont aláírások generálása
        
2.  **Biztonsági rendszer indítása**
    
    *   Valós idejű arcfelismerés aktiválása
        
    *   Automatikus ajtó nyitás/zárás
        
    *   Élő kamera kép felismerési állapottal
        
3.  **Rendszer beállítások**
    
    *   Felismerési küszöbérték beállítása (0.0-1.0)
        
    *   Nyitva tartási idő konfigurálása
        
    *   Kamera és Arduino beállítások
        
    *   Biztonsági paraméterek
        
4.  **Rendszer információk**
    
    *   Rendszer konfiguráció megjelenítése
        
    *   Kamera és port észlelési állapot
        
    *   Platform információ
        
5.  **Adatbázis kezelés**
    
    *   Regisztrált személyek listázása
        
    *   Felhasználók törlése
        
    *   Adatbázis biztonsági mentés és visszaállítás
        
    *   Statisztikák megtekintése
        
6.  **Nyelv váltás**
    
    *   Váltás magyar és angol nyelv között
        
7.  **Kilépés**
    
    *   Biztonságos rendszer leállítás
        

Technikai architektúra
----------------------

### Felismerési rendszer

A FaceGate kettős felismerési architektúrát használ:

1.  **CNN (Convolutional Neural Network)**
    
    *   Mély tanulás alapú arcreprezentációk
        
    *   128-dimenziós beágyazások
        
    *   Magas szintű funkcionalitás
        
2.  **LBPH (Local Binary Patterns Histograms)**
    
    *   Hagyományos arcfelismerési algoritmus
        
    *   Szürkeárnyalatos textúra elemzés
        
    *   Megbízható alacsony fényviszonyok között
        
3.  **Titkosított jellemzőpontok**
    
    *   Arc strukturális pontjainak detektálása
        
    *   XOR titkosítás
        
    *   Véletlenszerű biztonsági pontok
        

### Biztonsági rendszer

*   **Adattitkosítás**: Minden biometrikus adat XOR titkosítással van tárolva
    
*   **Kulcskezelés**: Automatikusan generált titkosítási kulcsok
    
*   **Adatbázis védelme**: Titkosított pickle formátum
    
*   **Minta minőségellenőrzés**: Automatikus minőségértékelés regisztráció közben
    

### Fájlstruktúra

FaceGate/

├── main.py # Fő program fájl

├── requirements.txt # Python függőségek

├── C\_UNLOCK.cpp # Arduino forráskód

├── data/

│ ├── secure\_facegate\_database.pkl

│ ├── encryption\_key.key

│ └── secure\_faces/

├── models/

│ └── secure\_facegate\_lbph\_model.xml

├── backups/

├── logs/

└── README.mdArduino konfiguráció

Csatlakozás

Csatlakoztasd az Arduino Uno-t a számítógéphez USB-n keresztül

Töltsd fel a C\_UNLOCK.cpp fájlt az Arduino IDE segítségével

A FaceGate rendszer automatikusan felismeri az Arduinót

Kommunikációs protokoll

Baud rate: 9600

Parancsok:

*   UNLOCK - Ajtó kinyitása
    
*   LOCK - Ajtó bezárása
    

### Hibaelhárítás

### Gyakori problémák

**Kamera nem észlelhető**

*   Ellenőrizd, hogy a kamera csatlakoztatva van-e
    
*   Más alkalmazás ne használja a kamerát
    
*   Linux rendszeren ellenőrizd a kamera jogosultságokat
    

**Arduino kapcsolati problémák**

*   Ellenőrizd a helyes COM port beállítást
    
*   Győződj meg róla, hogy az Arduino driver telepítve van
    
*   Ellenőrizd a soros kommunikációs beállításokat
    

**Felismerési pontossági problémák**

*   Állítsd be a küszöbértéket a rendszer beállításokban
    
*   Biztosíts megfelelő megvilágítást regisztráció közben
    
*   Regisztráld újra a felhasználókat jobb minőségű mintákkal
    

### Naplózás

A rendszer részletes naplókat generál a logs/ könyvtárban. A naplófájlok segítenek a hibakeresésben és a rendszer állapotának monitorozásában.

Fejlesztői információk
----------------------

**Fejlesztő**: Damjan Aros - THE PTI
**Projekt**: Egyetemi Projekt - FaceGate
**Egyetem**: Tokaj-Hegyalja Egyetem
**Kurzus**: Robotika, Mesterséges Intelligencia & Neurális Hálózatok
**Témavezetők**: Attila Perlaki, Dávid Gegény
**Félév**: 2025 Ősz

### Licenc
------

Ez a projekt oktatási célokra készült a Tokaj-Hegyalja Egyetem Robotika, MI & NN kurzusán. A kód felhasználása csak oktatási és kutatási célokra engedélyezett.

### Támogatás
---------

Problémák esetén ellenőrizd a rendszernaplókat és győződj meg arról, hogy minden függőség helyesen telepítve van. Hardveres problémák esetén ellenőrizd a kamera és Arduino csatlakozásokat.
