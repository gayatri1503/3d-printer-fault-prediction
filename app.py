from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load the trained model
model_path = 'model.pkl'
if os.path.exists(model_path):
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    print("Model loaded successfully!")
else:
    print("Model file not found! Please run the training notebook first.")
    model_data = None

@app.route('/')
def home():
    """Home page with input form"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests from web form"""
    try:
        if model_data is None:
            return render_template('result.html', 
                                 error='Model not loaded. Please train the model first.')
        
        # Get input data from form
        x_direction = float(request.form['x_direction'])
        y_direction = float(request.form['y_direction'])
        z_direction = float(request.form['z_direction'])
        
        # Validating input ranges 
        if not (-5 <= x_direction <= 5):
            raise ValueError("X-direction should be between -5 and 5")
        if not (-2 <= y_direction <= 8):
            raise ValueError("Y-direction should be between -2 and 8")
        if not (-15 <= z_direction <= 0):
            raise ValueError("Z-direction should be between -15 and 0")
        
        # Preparing input for prediction
        input_data = np.array([[x_direction, y_direction, z_direction]])
        
        # Scaling the input data
        scaled_input = model_data['scaler'].transform(input_data)
        
        # Making prediction
        prediction = model_data['model'].predict(scaled_input)[0]
        
        confidence = 85.0  
        if hasattr(model_data['model'], 'predict_proba'):
            try:
                prediction_proba = model_data['model'].predict_proba(scaled_input)[0]
                confidence = max(prediction_proba) * 100
            except:
                pass
        
        # Interpreting prediction
        result = "Error Detected" if prediction == 1 else "No Error"
        status = "danger" if prediction == 1 else "success"
        
        # Generating recommendations based on prediction
        if prediction == 1:
            recommendations = [
                "Check printer calibration and bed leveling",
                "Verify nozzle temperature and material quality",
                "Inspect mechanical components for wear",
                "Review and adjust print speed settings",
                "Consider scheduled maintenance",
                "Check for loose belts or damaged components"
            ]
            tips = [
                "Monitor the print closely for the first few layers",
                "Consider using a different material or settings",
                "Ensure proper bed adhesion before starting"
            ]
        else:
            recommendations = [
                "Print parameters are within optimal range",
                "Current sensor readings indicate normal operation",
                "Continue monitoring during print process",
                "Maintain regular calibration schedule"
            ]
            tips = [
                "Your printer settings look good to proceed",
                "Continue with current configuration",
                "Regular maintenance will keep performance optimal"
            ]
        
        return render_template('result.html', 
                             result=result,
                             status=status,
                             confidence=round(confidence, 1),
                             x_direction=x_direction,
                             y_direction=y_direction,
                             z_direction=z_direction,
                             recommendations=recommendations,
                             tips=tips,
                             model_name=model_data.get('model_name', 'ML Model'))
    
    except ValueError as ve:
        return render_template('result.html', 
                             error=f"Input validation error: {str(ve)}")
    except Exception as e:
        return render_template('result.html', 
                             error=f"Prediction error: {str(e)}")

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for programmatic access"""
    try:
        if model_data is None:
            return jsonify({
                'error': 'Model not loaded',
                'success': False
            })
        
        data = request.get_json()
        
        if not data:
            return jsonify({
                'error': 'No JSON data provided',
                'success': False
            })
        
        x_direction = float(data.get('x_direction', 0))
        y_direction = float(data.get('y_direction', 0))
        z_direction = float(data.get('z_direction', 0))
        
        # Preparing and scaling input
        input_data = np.array([[x_direction, y_direction, z_direction]])
        scaled_input = model_data['scaler'].transform(input_data)
        prediction = model_data['model'].predict(scaled_input)[0]
        
        # confidence
        confidence = 85.0
        if hasattr(model_data['model'], 'predict_proba'):
            try:
                prediction_proba = model_data['model'].predict_proba(scaled_input)[0]
                confidence = max(prediction_proba) * 100
            except:
                pass
        
        return jsonify({
            'prediction': int(prediction),
            'result': "Error Detected" if prediction == 1 else "No Error",
            'confidence': round(confidence, 1),
            'input': {
                'x_direction': x_direction,
                'y_direction': y_direction,
                'z_direction': z_direction
            },
            'success': True
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        })

@app.route('/about')
def about():
    """About page with project information"""
    model_info = {}
    if model_data:
        model_info = {
            'model_name': model_data.get('model_name', 'Unknown'),
            'accuracy': f"{model_data.get('accuracy', 0)*100:.1f}%",
            'features': model_data.get('feature_names', []),
            'dataset_size': model_data.get('dataset_info', {}).get('total_samples', 'Unknown')
        }
    
    return render_template('about.html', model_info=model_info)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
