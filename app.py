# # app.py
# import pickle
# import numpy as np
# from flask import Flask, render_template, request, jsonify
# import os

# app = Flask(__name__)

# # Load the pre-trained model
# model_path = 'randomForestModel.pkl'
# if os.path.exists(model_path):
#     with open(model_path, 'rb') as f:
#         model = pickle.load(f)
# else:
#     raise FileNotFoundError(f"Model file not found at {model_path}")

# @app.route('/', methods=['GET'])
# def index():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         # Get input data from the form
#         input_data = [float(request.form[f'feature_{i}']) for i in range(10)]
        
#         # Convert to numpy array and reshape
#         input_array = np.array(input_data).reshape(1, -1)
        
#         # Make prediction
#         prediction = model.predict(input_array)
        
#         # Convert prediction to human-readable output
#         result = 'Attack Detected' if prediction[0] == 1 else 'Normal Traffic'
        
#         return render_template('result.html', prediction=result)
    
#     except Exception as e:
#         return render_template('error.html', error=str(e))

# if __name__ == '__main__':
#     app.run(debug=True)



# app.py
import pickle
import numpy as np
from flask import Flask, render_template, request, jsonify
import os

app = Flask(__name__)

# Load the pre-trained model
model_path = 'randomForestModel.pkl'
if os.path.exists(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
else:
    raise FileNotFoundError(f"Model file not found at {model_path}")

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the form
        input_data = [
            float(request.form['feature_0']),  # Duration
            float(request.form['feature_1']),  # Protocol Type
            float(request.form['feature_2']),  # Service
            float(request.form['feature_3']),  # Flag Status
            float(request.form['feature_4']),  # Source Bytes
            float(request.form['feature_5']),  # Destination Bytes
            float(request.form['feature_6']),  # Land Connection
            float(request.form['feature_7']),  # Wrong Fragments
            float(request.form['feature_8']),  # Urgent Packets
            float(request.form['feature_9'])   # Hot Indicators
        ]
        
        # Convert to numpy array and reshape
        input_array = np.array(input_data).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(input_array)
        
        # Convert prediction to human-readable output
        result = 'Attack Detected' if prediction[0] == 1 else 'Normal Traffic'
        
        return render_template('result.html', prediction=result, input_data=input_data)
    
    except Exception as e:
        return render_template('error.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)