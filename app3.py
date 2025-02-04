import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, precision_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
@st.cache_data
def load_data():
    return pd.read_excel('/Users/ravindrababuponnapula/Downloads/US Insurance Claims Data.xlsx')

data = load_data()

# Prepare the data
X = data.drop(['fraud_reported'], axis=1)
y = data['fraud_reported']

# Handle missing values and encode target variable
data_cleaned = data.dropna()
X = data_cleaned.drop('fraud_reported', axis=1)
le = LabelEncoder()
y = le.fit_transform(data_cleaned['fraud_reported'])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define features
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# Create preprocessing steps
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('to_string', FunctionTransformer(lambda x: x.astype(str))),
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Create the preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create and train the models
models = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'XGBoost': XGBClassifier(random_state=42),
    'SVM': SVC(probability=True, random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42)
}

for name, model in models.items():
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    pipeline.fit(X_train, y_train)
    models[name] = pipeline

# Streamlit app
st.title('Insurance Fraud Detection')

# Input fields for user data
st.header('Enter Customer Information')
age = st.number_input('Age', min_value=18, max_value=100)
months_as_customer = st.number_input('Months as Customer', min_value=0)
policy_state = st.selectbox('Policy State', data['policy_state'].unique())
policy_csl = st.selectbox('Policy CSL', data['policy_csl'].unique())
policy_deductable = st.number_input('Policy Deductible', min_value=0)
policy_annual_premium = st.number_input('Policy Annual Premium', min_value=0.0)
umbrella_limit = st.number_input('Umbrella Limit', min_value=0)
insured_zip = st.number_input('Insured ZIP', min_value=0)
auto_make = st.selectbox('Auto Make', data['auto_make'].unique())
auto_model = st.selectbox('Auto Model', data['auto_model'].unique())
auto_year = st.number_input('Auto Year', min_value=1900, max_value=2050)
insured_education_level = st.selectbox('Education Level', data['insured_education_level'].unique())
insured_occupation = st.selectbox('Occupation', data['insured_occupation'].unique())
insured_hobbies = st.selectbox('Hobbies', data['insured_hobbies'].unique())
insured_relationship = st.selectbox('Relationship', data['insured_relationship'].unique())
incident_type = st.selectbox('Incident Type', data['incident_type'].unique())
collision_type = st.selectbox('Collision Type', data['collision_type'].unique())
incident_severity = st.selectbox('Incident Severity', data['incident_severity'].unique())
authorities_contacted = st.selectbox('Authorities Contacted', data['authorities_contacted'].unique())
incident_state = st.selectbox('Incident State', data['incident_state'].unique())
incident_city = st.selectbox('Incident City', data['incident_city'].unique())
incident_location = st.selectbox('Incident Location', data['incident_location'].unique())
incident_hour_of_the_day = st.number_input('Incident Hour', min_value=0, max_value=23)
number_of_vehicles_involved = st.number_input('Number of Vehicles Involved', min_value=1)
property_damage = st.selectbox('Property Damage', ['Yes', 'No'])
bodily_injuries = st.number_input('Bodily Injuries', min_value=0)
witnesses = st.number_input('Witnesses', min_value=0)
police_report_available = st.selectbox('Police Report Available', ['Yes', 'No'])
total_claim_amount = st.number_input('Total Claim Amount', min_value=0)
injury_claim = st.number_input('Injury Claim', min_value=0)
property_claim = st.number_input('Property Claim', min_value=0)
vehicle_claim = st.number_input('Vehicle Claim', min_value=0)

# Select model
selected_model = st.selectbox('Select Model', list(models.keys()))

insured_sex = st.selectbox('Insured Sex', data['insured_sex'].unique())
policy_number = st.number_input('Policy Number', min_value=0)
capital_gains = st.number_input('Capital Gains', min_value=0)
capital_loss = st.number_input('Capital Loss', min_value=0)

if st.button('Predict Fraud'):
    input_data = pd.DataFrame({
        'age': [age],
        'months_as_customer': [months_as_customer],
        'policy_number': [policy_number],
        'policy_state': [policy_state],
        'policy_csl': [policy_csl],
        'policy_deductable': [policy_deductable],
        'policy_annual_premium': [policy_annual_premium],
        'umbrella_limit': [umbrella_limit],
        'insured_zip': [insured_zip],
        'auto_make': [auto_make],
        'auto_model': [auto_model],
        'auto_year': [auto_year],
        'insured_education_level': [insured_education_level],
        'insured_occupation': [insured_occupation],
        'insured_hobbies': [insured_hobbies],
        'insured_relationship': [insured_relationship],
        'incident_type': [incident_type],
        'collision_type': [collision_type],
        'incident_severity': [incident_severity],
        'authorities_contacted': [authorities_contacted],
        'incident_state': [incident_state],
        'incident_city': [incident_city],
        'incident_location': [incident_location],
        'incident_hour_of_the_day': [incident_hour_of_the_day],
        'number_of_vehicles_involved': [number_of_vehicles_involved],
        'property_damage': [property_damage],
        'bodily_injuries': [bodily_injuries],
        'witnesses': [witnesses],
        'police_report_available': [police_report_available],
        'total_claim_amount': [total_claim_amount],
        'injury_claim': [injury_claim],
        'property_claim': [property_claim],
        'vehicle_claim': [vehicle_claim],
        'insured_sex': [insured_sex],
        'capital-gains': [capital_gains],
        'capital-loss': [capital_loss]
    })

    
    model = models[selected_model]
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)
    
    st.subheader('Prediction')
    st.write('Fraud Detected: Yes' if prediction[0] == 1 else 'Fraud Detected: No')
    st.write(f'Probability of Fraud: {probability[0][1]:.2%}')

# Model evaluation
st.subheader('Model Evaluation')
for name, model in models.items():
    y_pred = model.predict(X_test)
    st.write(f'{name} Performance:')
    st.text(classification_report(y_test, y_pred))
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    st.pyplot(plt)

st.sidebar.header('Insurance Fraud Detection')
st.sidebar.markdown('This app predicts the likelihood of insurance fraud based on customer information using multiple machine learning models.<br><br>**Team:**<br> Ravindra Babu Ponnapula<br>Akhil Karumanchi<br>Praveen Ravulapalli'
, unsafe_allow_html=True)


# Model evaluation
st.subheader('Model Evaluation')

# Create a list to store metrics for each model
metrics_data = []

for name, model in models.items():
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    metrics_data.append({
        'Model': name,
        'Accuracy': accuracy,
        'Precision': precision,
        'ROC AUC': roc_auc
    })
    

# Create a DataFrame from the metrics data and display it as a table
metrics_df = pd.DataFrame(metrics_data)
st.table(metrics_df.style.format({
    'Accuracy': '{:.2%}',
    'Precision': '{:.2%}',
    'ROC AUC': '{:.2f}'
}))
