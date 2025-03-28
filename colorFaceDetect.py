import cv2 as cv
import numpy as np

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
    visina, sirina, _ = slika.shape

    threshold = sirina_skatle * visina_skatle * 0.6
    
    skatle_matrika = []

    for y in range(0, visina, visina_skatle):
        vrstica = []
        for x in range(0, sirina, sirina_skatle):
            if x + sirina_skatle <= sirina and y + visina_skatle <= visina:
                skatla = slika[y:y + visina_skatle, x:x + sirina_skatle]
                stevilo_pikslov = prestej_piklse_z_barvo_koze(skatla, barva_koze)
                vrstica.append(1 if stevilo_pikslov > threshold else 0)

            else:
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

    #Zajami prvo sliko iz kamere

    #Izračunamo barvo kože na prvi sliki

    #Zajemaj slike iz kamere in jih obdeluj     
    
    #Označi območja (škatle), kjer se nahaja obraz (kako je prepuščeno vaši domišljiji)
        #Vprašanje 1: Kako iz števila pikslov iz vsake škatle določiti celotno območje obraza (Floodfill)?
        #Vprašanje 2: Kako prešteti število ljudi?

        #Kako velikost prebirne škatle vpliva na hitrost algoritma in točnost detekcije? Poigrajte se s parametroma velikost_skatle
        #in ne pozabite, da ni nujno da je škatla kvadratna.
    pass

