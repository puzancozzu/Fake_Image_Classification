# Django Implementation

# 1. Create a Django project and app:
# django-admin startproject myproject
# cd myproject
# python manage.py startapp myapp

# 2. Update settings.py to include 'myapp' in INSTALLED_APPS.

# 3. Replace views.py in myapp with the following code:


from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.http import HttpResponse
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
from smash_img import smash_n_reconstruct
from filters import apply_all_filters
import matplotlib.pyplot as plt
from io import BytesIO
import base64

from test_model import featureExtractionLayer

# Load the trained model
model_path = 'E:/Real_Fake_Image_Classification/Code/Notebooks/model_checkpoint_all_purpose.keras'
model = load_model(model_path, custom_objects={'featureExtractionLayer': featureExtractionLayer})

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Preprocessing function
def preprocess_image(path):
    try:
        rt, pt = smash_n_reconstruct(path)

        # Plot rich and poor textures
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(rt)
        ax[0].set_title('Rich Texture')
        ax[1].imshow(pt)
        ax[1].set_title('Poor Texture')
        plt.tight_layout()

        buffer = BytesIO()
        plt.savefig(buffer, format="png")
        buffer.seek(0)
        image_base64_1 = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close(fig)

        rt = apply_all_filters(rt)
        pt = apply_all_filters(pt)

        # Plot filtered textures
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(rt, cmap='gray')
        ax[0].set_title('Filtered Rich Texture')
        ax[1].imshow(pt, cmap='gray')
        ax[1].set_title('Filtered Poor Texture')
        plt.tight_layout()

        buffer = BytesIO()
        plt.savefig(buffer, format="png")
        buffer.seek(0)
        image_base64_2 = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close(fig)

        frt = tf.cast(tf.expand_dims(rt, axis=-1), dtype=tf.float64)
        fpt = tf.cast(tf.expand_dims(pt, axis=-1), dtype=tf.float64)
        frt = tf.ensure_shape(frt, [256, 256, 1])
        fpt = tf.ensure_shape(fpt, [256, 256, 1])
        frt = tf.expand_dims(frt, axis=0)
        fpt = tf.expand_dims(fpt, axis=0)
        
        return frt, fpt, image_base64_1, image_base64_2
    except Exception as e:
        print(f"Error processing {path}: {e}")
        dummy = tf.zeros([1, 256, 256, 1], dtype=tf.float32)
        return dummy, dummy, None, None

def index(request):
    if request.method == 'POST' and request.FILES['file']:
        file = request.FILES['file']
        fs = FileSystemStorage(location=UPLOAD_FOLDER)
        filename = fs.save(file.name, file)
        filepath = os.path.join(UPLOAD_FOLDER, filename)

        # Preprocess the image
        test_frt, test_fpt, image_base64_1, image_base64_2 = preprocess_image(filepath)

        # Predict
        predictions = model.predict([test_frt, test_fpt])

        # Interpret the result
        result = "Fake" if predictions > 0.5 else "Real"
        return render(request, 'result.html', {
            'prediction': result,
            'filename': filename,
            'image_base64_1': image_base64_1,
            'image_base64_2': image_base64_2
        })

    return render(request, 'index.html')

# 4. Create templates:
#   templates/index.html and templates/result.html as per Flask example.

# 5. Update urls.py to include:
# from django.urls import path
# from myapp import views
# urlpatterns = [
#     path('', views.index, name='index'),
# ]

# 6. Run the server:
# python manage.py runserver

