import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_data_generators(dataset_dir, img_size=128, batch_size=32, val_split=0.2):
    """
    Creates training and validation data generators using ImageDataGenerator.

    Args:
        dataset_dir (str): Path to the dataset directory.
        img_size (int): Image size to resize to (img_size x img_size).
        batch_size (int): Number of images per batch.
        val_split (float): Fraction of data to reserve for validation.

    Returns:
        train_gen, val_gen: Tuple of training and validation generators.
    """
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=val_split,
        rotation_range=20,
        zoom_range=0.2,
        shear_range=0.2,
        horizontal_flip=True
    )

    train_gen = datagen.flow_from_directory(
        dataset_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    val_gen = datagen.flow_from_directory(
        dataset_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )

    return train_gen, val_gen