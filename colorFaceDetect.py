import cv2 as cv
import numpy as np
import time as t

def zmanjsaj_sliko(slika, sirina, visina):
    slika = cv.resize(slika, (sirina, visina), interpolation=cv.INTER_AREA)
    return slika

def obdelaj_sliko_s_skatlami(slika, sirina_skatle, visina_skatle, barva_koze) -> list:
    '''Sprehodi se skozi sliko v velikosti škatle (sirina_skatle x visina_skatle) in izračunaj število pikslov kože v vsaki škatli.
    Škatle se ne smejo prekrivati!
    Vrne seznam škatel, s številom pikslov kože.
    Primer: Če je v sliki 25 škatel, kjer je v vsaki vrstici 5 škatel, naj bo seznam oblike
      [[1,0,0,1,1],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[1,0,0,0,1]]. 
      V tem primeru je v prvi škatli 1 piksel kože, v drugi 0, v tretji 0, v četrti 1 in v peti 1.'''
    # Pridobimo dimenzije slike
    visina, sirina, _ = slika.shape

    threshold = sirina_skatle * visina_skatle * 0.6  # 60% škatle mora biti koža
    
    # Seznam za shranjevanje rezultatov
    skatle_matrika = []

    # Prehajamo čez sliko v korakih velikosti škatle
    for y in range(0, visina, visina_skatle):
        vrstica = []  # Ena vrstica matrike
        for x in range(0, sirina, sirina_skatle):
            # Preverimo, da ne presežemo meje slike
            if x + sirina_skatle <= sirina and y + visina_skatle <= visina:
                # Izrežemo del slike (škatlo)
                skatla = slika[y:y + visina_skatle, x:x + sirina_skatle]
                # Preštejemo piksle z barvo kože v škatli
                stevilo_pikslov = prestej_piklse_z_barvo_koze(skatla, barva_koze)
                vrstica.append(1 if stevilo_pikslov > threshold else 0)

            else:
                # Če je škatla čez rob slike, dodamo 0 (ne upoštevamo)
                vrstica.append(0)
        skatle_matrika.append(vrstica)
    
    return skatle_matrika

def prestej_piklse_z_barvo_koze(slika, barva_koze) -> int:
    '''Prestej število pikslov z barvo kože v škatli.'''
    # Nastavimo dovoljeno odstopanje barve (toleranca)
    toleranca = 30
    
    # Določimo zgornjo in spodnjo mejo barve kože (BGR format)
    spodnja_meja = np.clip(np.array(barva_koze) - toleranca, 0, 255).astype(np.uint8)
    zgornja_meja = np.clip(np.array(barva_koze) + toleranca, 0, 255).astype(np.uint8)

    # Uporabimo masko za iskanje pikslov z barvo kože
    maska = cv.inRange(slika, spodnja_meja, zgornja_meja)
    
    # Preštejemo bele pike v maski (pikseli, ki so v določenem barvnem obsegu)
    stevilo_pikslov = cv.countNonZero(maska)
    
    return stevilo_pikslov

def doloci_barvo_koze(slika,levo_zgoraj,desno_spodaj) -> tuple:
    '''Ta funkcija se kliče zgolj 1x na prvi sliki iz kamere. 
    Vrne barvo kože v območju ki ga definira oklepajoča škatla (levo_zgoraj, desno_spodaj).
      Način izračuna je prepuščen vaši domišljiji.'''
    # Izrežemo pravokotno območje
    x1, y1 = levo_zgoraj
    x2, y2 = desno_spodaj
    roi = slika[y1:y2, x1:x2]  # Region of Interest (ROI)
    
    # Izračunamo povprečno barvo v območju
    povprecna_barva = np.mean(roi, axis=(0, 1))  # Srednja vrednost po vseh pikslov (po kanalih BGR)
    
    # Vrnemo barvo kot tuple v BGR formatu
    return tuple(map(int, povprecna_barva))

if __name__ == '__main__':
    #Pripravi kamero
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    #Zajami prvo sliko iz kamere
    ret, firstFrame = cap.read()
    if not ret:
        cap.release()
        cv.destroyAllWindows()
        print("Can't receive frame (stream end?). Exiting ...")
        exit()

    firstFrame = zmanjsaj_sliko(firstFrame, 260, 300)

    #Izračunamo barvo kože na prvi sliki
    barva = doloci_barvo_koze(firstFrame, (105,125), (155,175))

    prev_time = 0

    #Zajemaj slike iz kamere in jih obdeluj     
    while True:
        ret, frame = cap.read()
 
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        if cv.waitKey(1) == ord('q'):
            break

        frame = zmanjsaj_sliko(frame, 260, 300)

        #Označi območja (škatle), kjer se nahaja obraz (kako je prepuščeno vaši domišljiji)
        #Vprašanje 1: Kako iz števila pikslov iz vsake škatle določiti celotno območje obraza (Floodfill)?
        #Vprašanje 2: Kako prešteti število ljudi?
        boxSize = 10;
        obraz = obdelaj_sliko_s_skatlami(frame, boxSize, boxSize, barva)

        # Najdi min in max koordinate območja obraza
        min_x, min_y = float('inf'), float('inf')
        max_x, max_y = float('-inf'), float('-inf')

        for y, row in enumerate(obraz):
            for x, val in enumerate(row):
                if val == 1:
                    min_x = min(min_x, x)
                    min_y = min(min_y, y)
                    max_x = max(max_x, x)
                    max_y = max(max_y, y)

        # Nariši škatlo okoli obraza
        if min_x < float('inf') and min_y < float('inf'):
            top_left = (min_x * boxSize, min_y * boxSize)
            bottom_right = ((max_x + 1) * boxSize, (max_y + 1) * boxSize)
            cv.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)



        current_time = t.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time
        cv.putText(frame, f"FPS: {int(fps)}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv.imshow('frame', frame)

        #Kako velikost prebirne škatle vpliva na hitrost algoritma in točnost detekcije? Poigrajte se s parametroma velikost_skatle
        #in ne pozabite, da ni nujno da je škatla kvadratna.

    cap.release()
    cv.destroyAllWindows()
