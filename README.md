# Image-Classification-using-CNN-and-Transfer-Learning
The following Repository contains a deep learning-based web application for classifying different types of rice grains (e.g. Arborio, Basmati, Ipsala, Jasmine, Karacadag) using a trained Convolutional Neural Network (CNN) model. The model is trained using the rice image dataset with the image files. The image is preprocessed or normalized before uploading. Libraries like tensorflow, keras, sklearn.metrics used for load the model and precesion, recall and F1-score. The model is evaluated using the validation set of loss and accuracy. Streamlit is used for web deployment, html and static uploads files are used for frontend. 

structure:
Rice_Image_Dataset/
│
├── src/
│   ├── _pycache_/
│   ├── predict.py               # Contains predict_class function
│   └── preprocessing.py         # Contains preprocess_image function
│
├── static/                      
│
├── templates/
│   ├── index.html               # Upload or input page
│   └── result.html              # Displays prediction result
│
├── Image_Classification_using_CNN.ipynb   # Google Colab Notebook for training
├── main.py                      # Streamlit application entry point
├── model (2).h5                 # Trained CNN model


Environment setup and dependancies:-

Tensorflow, Keras, Scikit-learn, Numpy, Pillow install using:-
pip install tensorflow scikit-learn matplotlib seaborn Pillow streamlit    





