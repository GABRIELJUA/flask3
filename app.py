from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

# Cargar modelo y escalador
model = joblib.load('mlp_model.pkl')
scaler = joblib.load('scaler.pkl')
app.logger.debug('Modelo y escalador cargados correctamente.')

@app.route('/')
def home():
    return render_template('formulario.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Capturar los valores desde el formulario
        Duration = float(request.form['Duration'])
        Heart_Rate = float(request.form['Heart_Rate'])
        Age = float(request.form['Age'])
        Body_Temp = float(request.form['Body_Temp'])

        # Crear DataFrame con los valores
        data_df = pd.DataFrame([[Duration, Heart_Rate, Age, Body_Temp]],
                               columns=["Duration", "Heart_Rate", "Age", "Body_Temp"])
        app.logger.debug(f'DataFrame de entrada (sin escalar):\n{data_df}')

        # Aplicar escalado a los datos
        data_scaled = scaler.transform(data_df)
        app.logger.debug(f'Data escalada:\n{data_scaled}')

        # Hacer predicción con datos escalados
        prediction = model.predict(data_scaled)
        app.logger.debug(f'Predicción realizada: {prediction[0]}')

        return jsonify({'prediccion': f"{prediction[0]:.2f} gramos de grasa quemada"})
    except Exception as e:
        app.logger.error(f'Error en la predicción: {str(e)}')
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
