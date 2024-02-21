from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model and top 10 features
rf_classifier, top_10_features = joblib.load('income_prediction_model.pkl')

@app.route('/')
def home():
    return render_template('index.html', features=top_10_features['Feature'])

@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from the form
    input_data = {}
    for feature in top_10_features['Feature']:
        input_data[feature] = request.form[feature]

    # Convert input data into a DataFrame
    input_df = pd.DataFrame(input_data, index=[0])

    # Make prediction
    prediction = rf_classifier.predict(input_df)

    # Render the result template with the prediction
    return render_template('result.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
