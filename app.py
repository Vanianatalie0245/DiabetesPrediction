from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)


def clip_outliers(data, lower_percentile=0.25, upper_percentile=0.75, factor=1.5):
    df_clipped = data.copy()
    for column in df_clipped.columns:
        Q1 = df_clipped[column].quantile(lower_percentile)
        Q3 = df_clipped[column].quantile(upper_percentile)
        IQR = Q3 - Q1

        lower_bound = Q1 - (factor * IQR)
        upper_bound = Q3 + (factor * IQR)

        df_clipped[column] = np.clip(df_clipped[column], lower_bound, upper_bound)
    
    return df_clipped

model = joblib.load('final_rff_model.pkl')
preprocessing_pipeline = joblib.load('preprocessing_pipeline1.pkl')

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        name = request.form.get('name') 
        glucose = float(request.form.get('glucose'))
        bmi = float(request.form.get('bmi'))
        blood_pressure = float(request.form.get('bloodPressure'))
        insulin = float(request.form.get('insulin'))
        pregnancy = int(request.form.get('pregnancy'))
        skin_thickness = float(request.form.get('skinThickness'))
        age = int(request.form.get('age'))

        input_data = pd.DataFrame([{
            'Pregnancies': pregnancy,
            'Glucose': glucose,
            'BloodPressure': blood_pressure,
            'SkinThickness': skin_thickness,
            'Insulin': insulin,
            'BMI': bmi,
            'Age': age
        }], columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'Age'])
        


        preprocessed_data = preprocessing_pipeline.transform(input_data)
        prediction = model.predict(preprocessed_data)[0]

        return jsonify({'prediction': int(prediction)})
    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)



