from flask import Flask, render_template, request
import numpy as np
from PIL import Image
from keras.models import load_model
from keras.applications.mobilenet_v2 import preprocess_input

# Load the trained model
model = load_model('tea_leaf_model.h5')

# Initialize Flask application
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/leaf')
def leaf():
    return render_template('leaf.html')

# Define route for home page
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the image file from the request
        file = request.files['image']

        # Read the image file
        img = Image.open(file)
        img = img.resize((224, 224))
        
        # Preprocess the image
        img = np.array(img)
        img = preprocess_input(img)
        img = np.expand_dims(img, axis=0)

        # Make the prediction
        prediction = model.predict(img)
        class_labels = ['Anthracnose', 'algal leaf', 'bird eye spot', 'brown blight', 'gray light', 'healthy', 'red leaf spot', 'white spot']
        predicted_class = class_labels[np.argmax(prediction)]

        return render_template('predict.html', prediction=predicted_class)
    
    return render_template('predict.html')

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=False, port=8080)
