import cv2
import numpy as np
from keras.models import load_model

# Charger le modèle Keras
model = load_model("facialemotionmodel.h5")

# Charger le fichier de cascade haar pour la détection de visage
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Définir un dictionnaire qui mappe les indices de classes aux émotions
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

# Définir un dictionnaire qui mappe les émotions aux conseils
conseils = {
    'angry': "Respirez profondement et prenez une pause pour vous calmer.",
    'disgust': "Identifiez ce qui vous derange et essayez de trouver une solution.",
    'fear': "Essayez de vous concentrer sur le moment present et pratiquez la respiration profonde.",
    'happy': "Celebrez vos reussites et partagez votre bonheur avec les autres.",
    'neutral': "Prenez une pause et reflechissez a ce qui vous rend heureux.",
    'sad': "Exprimez vos emotions a un proche et pratiquez la gratitude.",
    'surprise': "Prenez le temps de vous adapter a la nouvelle situation et restez ouvert aux opportunités."
}

# Définir une fonction pour extraire les caractéristiques de l'image
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# Capturer la vidéo depuis la webcam
webcam = cv2.VideoCapture(0)

while True:
    # Lire une image depuis la webcam
    i, im = webcam.read()
    
    # Vérifier si l'image a été correctement lue
    if im is not None:
        # Convertir l'image en niveaux de gris
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        
        # Détecter les visages dans l'image
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        # Parcourir les visages détectés
        for (p, q, r, s) in faces:
            # Extraire le visage de l'image en niveaux de gris
            face_image = gray[q:q+s, p:p+r]
            
            # Dessiner un rectangle autour du visage sur l'image originale
            cv2.rectangle(im, (p, q), (p+r, q+s), (255, 0, 0), 2)
            
            # Redimensionner l'image du visage à la taille attendue par le modèle
            face_image = cv2.resize(face_image, (48, 48))
            
            # Extraire les caractéristiques du visage pour la prédiction
            features = extract_features(face_image)
            
            # Faire la prédiction d'émotion
            pred_emotion = model.predict(features)
            
            # Obtenir l'étiquette prédite de l'émotion
            emotion_label = labels[pred_emotion.argmax()]
            
            # Obtenir le conseil correspondant à l'émotion détectée
            conseil = conseils.get(emotion_label, "Pas de conseil disponible pour cette émotion.")
            
            # Afficher l'émotion prédite sur l'image
            cv2.putText(im, emotion_label, (p-10, q-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255))
            
            # Afficher le conseil sur plusieurs lignes si nécessaire
            lines = [conseil[i:i+30] for i in range(0, len(conseil), 30)]
            for idx, line in enumerate(lines):
                cv2.putText(im, line, (p-10, q-10 + (idx+1)*20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (0, 0, 255))
        
        # Afficher l'image avec les visages détectés et les étiquettes prédites
        cv2.imshow("Output", im)
        
        # Attendre l'appui sur la touche 'Esc' pour quitter
        if cv2.waitKey(1) == 27:
            break

# Libérer la capture vidéo et fermer toutes les fenêtres
webcam.release()
cv2.destroyAllWindows()
