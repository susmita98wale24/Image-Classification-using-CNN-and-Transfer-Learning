from PIL import Image
from io import BytesIO
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

model = load_model('E:/Image_classification_system/model (2).h5')
class_names = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']

def predict_rice_type(pil_img: Image.Image) -> str:
    # Convert PIL image to BytesIO
    img_bytes = BytesIO()
    pil_img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)

    # Load and preprocess image
    img = image.load_img(img_bytes, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    return class_names[np.argmax(prediction)]