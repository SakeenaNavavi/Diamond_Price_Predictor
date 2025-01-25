from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_regression
import joblib
import os

app = Flask(__name__)

class DiamondPricePredictor:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_selectors = {}
        self.encoders = {}
        self.feature_columns = [
            'Carat Weight', 'Cut', 'Color', 'Clarity', 'Length', 'Width', 'Height',
            'Length/Width Ratio', 'Depth %', 'Table %', 'Culet', 'Fluorescence', 
            'Girdle', 'Polish', 'Price_per_carat', 'Shape', 'Symmetry', 'Type', 'Volume'
        ]
        
        self.load_models()
    
    def load_models(self):
        models_path = 'models/'
        try:
            self.models = {
                'Linear Regression': joblib.load(f'{models_path}linear_regression.pkl'),
                'Random Forest': joblib.load(f'{models_path}random_forest.pkl'),
                'XGBoost': joblib.load(f'{models_path}xgboost.pkl'),
                'SVR': joblib.load(f'{models_path}svr.pkl'),
                'KNN': joblib.load(f'{models_path}knn.pkl')
            }
            self.scaler = joblib.load(f'{models_path}scaler.pkl')
            self.feature_selector = joblib.load(f'{models_path}feature_selector.pkl')
            self.encoders = joblib.load(f'{models_path}encoders.pkl')
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            raise
    
    def handle_unseen_labels(input_data, encoder, col_name, default_value='Fair'):
        if input_data[col_name] not in encoder.classes_:
            print(f"Warning: '{input_data[col_name]}' is an unseen label. Using default: '{default_value}'.")
            input_data[col_name] = default_value  # Set to default value if unseen
        return encoder.transform([input_data[col_name]])[0]


    def predict(self, input_data):
        try:
            df = pd.DataFrame([input_data], columns=self.feature_columns)
            
            for col in ['Cut', 'Color', 'Clarity', 'Shape', 'Symmetry', 'Type']:
                if pd.isnull(input_data[col]) or input_data[col] == 'None':
                    return {"error": f"Value for {col} is missing or invalid. Please provide a valid value."}
                
                if col == 'Shape' and input_data[col] not in self.encoders[col].classes_:
                  
                    input_data[col] = 'Round' 
                    df[col] = self.encoders[col].transform([input_data[col]])
                else:
                    if col in self.encoders:
                        df[col] = self.encoders[col].transform(df[col])
            
            df['Volume'] = df['Length'] * df['Width'] * df['Height']
            
            features_selected = self.feature_selector.transform(df)
        
            features_scaled = self.scaler.transform(features_selected)
            
            predictions = {}
            for name, model in self.models.items():
                pred = model.predict(features_scaled)[0]
                predictions[name] = round(pred, 2)
            
            predictions['Average'] = round(sum(predictions.values()) / len(predictions), 2)
            
            return predictions
            
        except Exception as e:
            print(f"Error making prediction: {str(e)}")
            return {"error": str(e)}



predictor = DiamondPricePredictor()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = {
            'Carat Weight': float(request.form.get('carat', 1.0)),  # Default value if missing
            'Cut': request.form.get('cut', 'Fair'),
            'Color': request.form.get('color', 'D'),
            'Clarity': request.form.get('clarity', 'SI1'),
            'Length': float(request.form.get('length', 6.0)),
            'Width': float(request.form.get('width', 6.0)),
            'Height': float(request.form.get('height', 4.0)),
            'Length/Width Ratio': float(request.form.get('length_width_ratio', 1.0)),
            'Depth %': float(request.form.get('depth_percent', 61.0)),
            'Table %': float(request.form.get('table_percent', 57.0)),
            'Culet': request.form.get('culet', 'None'),
            'Fluorescence': request.form.get('fluorescence', 'None'),
            'Girdle': request.form.get('girdle', 'None'),
            'Polish': request.form.get('polish', 'None'),
            'Price_per_carat': None,
            'Shape': request.form.get('shape', 'Round'),
            'Symmetry': request.form.get('symmetry', 'Fair'),
            'Type': request.form.get('type', 'Ideal'),
        }


        print(f"Input data: {input_data}")

        
        for col in ['Shape', 'Symmetry', 'Type']:
            if input_data[col] is not None:
                input_data[col] = predictor.encoders[col].transform([input_data[col]])[0]
        

        input_data['Shape'] = predictor.handle_unseen_labels(input_data, predictor.encoders['Shape'], 'Shape', default_value='Round')
        input_data['Symmetry'] = predictor.handle_unseen_labels(input_data, predictor.encoders['Symmetry'], 'Symmetry', default_value='Fair')
        input_data['Type'] = predictor.handle_unseen_labels(input_data, predictor.encoders['Type'], 'Type', default_value='Ideal')


        predictions = predictor.predict(input_data)
        
        return jsonify({
            'success': True,
            'predictions': predictions
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })



if __name__ == '__main__':
    app.run(debug=True)