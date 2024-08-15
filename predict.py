import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import sys



def predict_leaf_health(image_path):
    input_shape = (128, 128, 3)
    batch_size = 32
    epochs = 3
    num_classes = 38
    model = load_model('plant_disease_detection_multiclass_model.h5')

    img = image.load_img(image_path, target_size=(128, 128)) 
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]

    train_directory = r'archive\PlantVillage\train'  
    validation_directory = r'archive\PlantVillage\val'


    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    validation_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_directory,
        target_size=(128, 128),
        batch_size=batch_size,
        class_mode='categorical'  # Multi-class classification
    )

    validation_generator = validation_datagen.flow_from_directory(
        validation_directory,
        target_size=(128, 128),
        batch_size=batch_size,
        class_mode='categorical'  # Multi-class classification
    )


    class_labels = list(validation_generator.class_indices.keys())  
    predicted_label = class_labels[predicted_class]

    return predicted_label 



