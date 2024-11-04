import tensorflow as tf
import cv2
import numpy as np

# Carregar modelo
model = tf.keras.models.load_model('models/face_mask_model.keras')

def detect_mask(frame, threshold=0.4):  # Adicionando threshold ajustável
    # Pré-processar a imagem
    face_img = cv2.resize(frame, (128, 128))
    face_img = np.expand_dims(face_img, axis=0) / 255.0

    # Prever máscara ou não
    prediction = model.predict(face_img)
    label = "Mask" if prediction < threshold else "No Mask"
    
    # Exibir o resultado
    color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    return frame
