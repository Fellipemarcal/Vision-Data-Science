import os
import sys
import time
import numpy as np
import cv2

cap = cv2.VideoCapture('MediaPipe_et_OpenCV/Alarme/video.mp4')
# cap = cv2.VideoCapture("chien.mp4")

kernel_blur = 5
seuil = 15
surface = 1000
ret, originale = cap.read()
originale = cv2.cvtColor(originale, cv2.COLOR_BGR2GRAY)
originale = cv2.GaussianBlur(originale, (kernel_blur, kernel_blur), 0)
kernel_dilate = np.ones((5, 5), np.uint8)

# Initialisation des compteurs
object_count = 0
person_count = 0

# Liste pour stocker les objets détectés (par leur bounding box)
detected_objects = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (kernel_blur, kernel_blur), 0)
    
    mask = cv2.absdiff(originale, gray)
    mask = cv2.threshold(mask, seuil, 255, cv2.THRESH_BINARY)[1]
    mask = cv2.dilate(mask, kernel_dilate, iterations=3)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame_contour = frame.copy()

    # Liste pour les objets détectés dans cette frame
    current_frame_objects = []

    for c in contours:
        cv2.drawContours(frame_contour, [c], 0, (0, 255, 0), 5)
        
        if cv2.contourArea(c) < surface:
            continue
        
        # Calculer le bounding box de l'objet
        x, y, w, h = cv2.boundingRect(c)
        
        # Ajouter l'objet à la liste des objets détectés dans cette frame
        current_frame_objects.append((x, y, w, h))
        
        # Comptabilisation des objets
        object_count += 1
        
        # Détection simplifiée des personnes (objets plus grands)
        if cv2.contourArea(c) > 2000:  # Ajustez cette valeur pour mieux détecter les personnes
            person_count += 1
        
        # Dessiner le rectangle autour de l'objet
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Comparer les objets détectés avec ceux de la frame précédente
    new_detected_objects = []
    for obj in current_frame_objects:
        # Ajouter les objets de cette frame à la liste finale s'ils ne sont pas déjà comptabilisés
        if obj not in detected_objects:
            new_detected_objects.append(obj)
    
    # Mettre à jour la liste des objets détectés
    detected_objects = new_detected_objects

    originale = gray

    # Afficher le comptage des objets et des personnes
    cv2.putText(frame, f"Objets détectés: {len(detected_objects)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(frame, f"Personnes détectées: {person_count}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
    cv2.putText(frame, "[o|l]seuil: {:d}  [p|m]blur: {:d}  [i|k]surface: {:d}".format(seuil, kernel_blur, surface),
                (10, 90), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 255), 2)
    
    cv2.imshow("frame", frame)
    cv2.imshow("contour", frame_contour)
    cv2.imshow("mask", mask)
    
    intrus = 0
    key = cv2.waitKey(30) & 0xFF
    if key == ord('q'):
        break
    if key == ord('p'):
        kernel_blur = min(43, kernel_blur + 2)
    if key == ord('m'):
        kernel_blur = max(1, kernel_blur - 2)
    if key == ord('i'):
        surface += 1000
    if key == ord('k'):
        surface = max(1000, surface - 1000)
    if key == ord('o'):
        seuil = min(255, seuil + 1)
    if key == ord('l'):
        seuil = max(1, seuil - 1)

cap.release()
cv2.destroyAllWindows()
