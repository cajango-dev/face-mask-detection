import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # Saída binária: com ou sem máscara
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train():
    # Carregar dataset
    datagen = ImageDataGenerator(rescale=1.0/255, validation_split=0.2)
    train_gen = datagen.flow_from_directory(
        'dataset',
        target_size=(128, 128),
        batch_size=32,
        class_mode='binary',
        subset='training'
    )
    val_gen = datagen.flow_from_directory(
        'dataset',
        target_size=(128, 128),
        batch_size=32,
        class_mode='binary',
        subset='validation'
    )

    model = create_model()
    model.fit(train_gen, validation_data=val_gen, epochs=10)
    model.save('models/face_mask_model.keras')

if __name__ == "__main__":
    train()
