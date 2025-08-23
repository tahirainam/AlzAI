from flask import Flask, render_template, request
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
import shap
import matplotlib
matplotlib.use('Agg')   # Use non-GUI backend to avoid Tkinter errors
import matplotlib.pyplot as plt
import base64
from io import BytesIO

app = Flask(__name__)

# Load trained model
model = joblib.load('model/best_model.pkl')

# Initialize scaler (in production, load the scaler used in training)
scaler = MinMaxScaler()

# Feature descriptions for the 15 selected features
feature_descriptions = {
    'MemoryComplaints': '0 = No, 1 = Yes',
    'BehavioralProblems': '0 = No, 1 = Yes',
    'FunctionalAssessment': 'Score measuring daily functioning',
    'ADL': 'Activities of Daily Living score',
    'MMSE': 'Mini-Mental State Examination score (0-30)',
    'Hypertension': '0 = No, 1 = Yes',
    'CardiovascularDisease': '0 = No, 1 = Yes',
    'Diabetes': '0 = No, 1 = Yes',
    'FamilyHistoryAlzheimers': '0 = No, 1 = Yes',
    'SleepQuality': 'Sleep quality rating (1-5)',
    'Disorientation': '0 = No, 1 = Yes',
    'HeadInjury': '0 = No, 1 = Yes',
    'EducationLevel': 'Years of education completed',
    'PersonalityChanges': '0 = No, 1 = Yes',
    'CholesterolHDL': 'HDL cholesterol level (mg/dL)'
}

# Features actually used in training (15 selected features)
selected_features = [
    'MemoryComplaints', 'BehavioralProblems', 'FunctionalAssessment', 'ADL',
    'MMSE', 'Hypertension', 'CardiovascularDisease', 'Diabetes',
    'FamilyHistoryAlzheimers', 'SleepQuality', 'Disorientation', 'HeadInjury',
    'EducationLevel', 'PersonalityChanges', 'CholesterolHDL'
]

@app.route('/')
def home():
    return render_template('index.htm', features=feature_descriptions)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect form data
        form_data = request.form.to_dict()

        # Convert to DataFrame with correct column order
        input_data = pd.DataFrame([form_data])[selected_features]

        

        # Convert numeric fields
        numeric_cols = ['FunctionalAssessment', 'ADL', 'MMSE',
                        'SleepQuality', 'EducationLevel', 'CholesterolHDL']
        for col in numeric_cols:
            input_data[col] = pd.to_numeric(input_data[col], errors='coerce')

        # Convert binary fields
        binary_cols = ['MemoryComplaints', 'BehavioralProblems',
                       'Hypertension', 'CardiovascularDisease', 'Diabetes',
                       'FamilyHistoryAlzheimers', 'Disorientation',
                       'HeadInjury', 'PersonalityChanges']
        for col in binary_cols:
            input_data[col] = input_data[col].map({'0': 0, '1': 1, 'no': 0, 'yes': 1}).astype(int)

        # Scale data
        X_scaled = pd.DataFrame(scaler.fit_transform(input_data), columns=input_data.columns)

        # Prediction
        prediction = model.predict(X_scaled)[0]
        probability = model.predict_proba(X_scaled)[0]

        # SHAP explanation
        explainer = shap.Explainer(model, X_scaled)
        shap_values = explainer(X_scaled, check_additivity=False)

        # SHAP bar plot
        plt.figure()
        shap.summary_plot(shap_values, X_scaled, plot_type="bar", show=False)
        plt.tight_layout()

        # Save SHAP plot to base64
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        shap_plot = base64.b64encode(buf.getvalue()).decode('ascii')

        # Results
        diagnosis = "Alzheimer's Disease Detected" if prediction == 1 else "No Alzheimer's Disease Detected"
        confidence = round(max(probability) * 100, 2)

        return render_template(
            'result.htm',
            diagnosis=diagnosis,
            confidence=confidence,
            shap_plot=shap_plot,
            input_data=form_data,
            feature_descriptions=feature_descriptions
        )

    except Exception as e:
        return render_template('index.htm', features=feature_descriptions, error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
