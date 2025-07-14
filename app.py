from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import numpy as np
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

# Cargar modelo y escalador
model = joblib.load('random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')
ordinal_encoder = joblib.load('encoder.pkl')
pca = joblib.load('pca.pkl')
category_options = joblib.load('category_options.pkl')
app.logger.debug('Modelos, encoder, PCA y categorías cargados correctamente.')

@app.route('/')
def home():
    return render_template('formulario.html')

@app.route('/get_categories')
def get_categories():
    return jsonify(category_options)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener los datos enviados en el formulario
        data = {
            'Pclass': float(request.form['Pclass']),
            'Sex': request.form['Sex'],
            'Age': float(request.form['Age']),
            'SibSp': float(request.form['SibSp']),
            'Parch': float(request.form['Parch']),
            'Ticket': request.form['Ticket'],
            'Fare': float(request.form['Fare']),
            'Cabin': request.form['Cabin'],
            'Embarked': request.form['Embarked']
        }

        # Validar entradas numéricas
        if data['Pclass'] not in [1, 2, 3]:
            raise ValueError("La clase (Pclass) debe ser 1, 2 o 3.")
        if data['Age'] < 0 or data['Age'] > 100:
            raise ValueError("La edad debe estar entre 0 y 100 años.")
        if data['SibSp'] < 0 or data['SibSp'] > 10:
            raise ValueError("El número de hermanos/cónyuges debe estar entre 0 y 10.")
        if data['Parch'] < 0 or data['Parch'] > 10:
            raise ValueError("El número de padres/hijos debe estar entre 0 y 10.")
        if data['Fare'] < 0:
            raise ValueError("La tarifa no puede ser negativa.")

        # Verificar que las categorías existan en category_options
        if data['Sex'] not in category_options['Sex']:
            raise ValueError(f"El sexo debe ser uno de {category_options['Sex']}.")
        if data['Embarked'] not in category_options['Embarked']:
            raise ValueError(f"El puerto de embarque debe ser uno de {category_options['Embarked']}.")
        if data['Ticket'] not in category_options['Ticket']:
            data['Ticket'] = category_options['Ticket'][0]  # Valor por defecto
        if data['Cabin'] not in category_options['Cabin']:
            data['Cabin'] = 'Desconocido'  # Valor por defecto

        # Crear un DataFrame con los datos
        data_df = pd.DataFrame([data], columns=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'])
        app.logger.debug(f'DataFrame creado: {data_df}')

        # Transformar las variables categóricas con el OrdinalEncoder
        categorical_cols = ['Sex', 'Embarked', 'Ticket', 'Cabin']
        data_array = [data_df[categorical_cols].values[0]]  # Convertir a lista de un solo array
        encoded_values = ordinal_encoder.transform(data_array)
        data_df[categorical_cols] = encoded_values
        app.logger.debug(f'DataFrame tras encoding: {data_df}')

        # Escalar los datos
        X_scaled = scaler.transform(data_df)
        app.logger.debug(f'Datos escalados: {X_scaled}')

        # Aplicar PCA
        X_pca = pca.transform(X_scaled)
        app.logger.debug(f'Datos tras PCA: {X_pca}')

        # Realizar predicción
        prediction = model.predict(X_pca)[0]
        result = 'Sobrevivió' if prediction == 1 else 'No sobrevivió'
        app.logger.debug(f'Predicción: {result}')

        # Devolver la predicción como respuesta JSON
        return jsonify({'result': result})
    except Exception as e:
        app.logger.error(f'Error en la predicción: {str(e)}')
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
