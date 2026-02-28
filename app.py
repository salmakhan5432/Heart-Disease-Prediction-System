import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for responsive design
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #e74c3c;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .prediction-result {
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    
    .healthy {
        background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%);
        color: white;
    }
    
    .at-risk {
        background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
        color: white;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
        padding: 10px 0;
        font-size: 0.9rem;
        z-index: 999;
    }
    
    .made-by {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 2rem 0;
        font-size: 1.1rem;
        font-weight: bold;
    }
    
    @media (max-width: 768px) {
        .main-header {
            font-size: 2rem;
        }
        .prediction-result {
            font-size: 1.2rem;
            padding: 1rem;
        }
        .footer {
            font-size: 0.8rem;
            padding: 8px 0;
        }
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and preprocess the heart disease dataset"""
    df = pd.read_csv('heart_disease_uci.csv')
    
    # Create binary target variable
    df['target'] = df['num'].apply(lambda x: 0 if x == 0 else 1)
    
    # Handle missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    
    categorical_cols = df.select_dtypes(include='object').columns
    for col in categorical_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)
    
    # Remove duplicates
    df.drop_duplicates(inplace=True)
    
    return df

@st.cache_data
def prepare_model_data(df):
    """Prepare data for machine learning models"""
    # Select relevant features
    features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                'thalch', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    
    # Create a copy for modeling
    model_df = df[features + ['target']].copy()
    
    # Encode categorical variables
    le_dict = {}
    categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']
    
    for col in categorical_features:
        if col in model_df.columns:
            le = LabelEncoder()
            model_df[col] = le.fit_transform(model_df[col].astype(str))
            le_dict[col] = le
    
    return model_df, le_dict

def train_models(X_train, X_test, y_train, y_test):
    """Train multiple models and return results"""
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    results = {}
    trained_models = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        results[name] = {
            'accuracy': accuracy,
            'predictions': y_pred,
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        trained_models[name] = model
    
    return results, trained_models

def main():
    # Header
    st.markdown('<h1 class="main-header">❤️ Heart Disease Prediction Dashboard</h1>', 
                unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    model_df, le_dict = prepare_model_data(df)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", 
                               ["🏠 Home", "📊 Data Analysis", "🤖 Prediction", "📈 Model Performance"])
    
    if page == "🏠 Home":
        show_home_page(df)
    elif page == "📊 Data Analysis":
        show_analysis_page(df)
    elif page == "🤖 Prediction":
        show_prediction_page(model_df, le_dict)
    elif page == "📈 Model Performance":
        show_model_performance_page(model_df)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem 0;">
        <p>❤️ Made with Streamlit by <strong>Salma Khan</strong> | Heart Disease Prediction System</p>
    </div>
    """, unsafe_allow_html=True)

def show_home_page(df):
    """Display the home page with overview"""
    st.markdown("## Welcome to the Heart Disease Prediction System")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{len(df)}</h3>
            <p>Total Patients</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        heart_disease_count = df['target'].sum()
        st.markdown(f"""
        <div class="metric-card">
            <h3>{heart_disease_count}</h3>
            <p>With Heart Disease</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        healthy_count = len(df) - heart_disease_count
        st.markdown(f"""
        <div class="metric-card">
            <h3>{healthy_count}</h3>
            <p>Healthy Patients</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        avg_age = df['age'].mean()
        st.markdown(f"""
        <div class="metric-card">
            <h3>{avg_age:.1f}</h3>
            <p>Average Age</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Dataset overview
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Dataset Overview")
        st.markdown("""
        This application analyzes heart disease data from the UCI Heart Disease dataset. 
        The dataset contains various medical attributes that can help predict the presence 
        of heart disease in patients.
        
        **Key Features:**
        - **Age**: Patient's age in years
        - **Sex**: Patient's gender
        - **Chest Pain Type**: Type of chest pain experienced
        - **Blood Pressure**: Resting blood pressure
        - **Cholesterol**: Serum cholesterol level
        - **Blood Sugar**: Fasting blood sugar > 120 mg/dl
        - **ECG Results**: Resting electrocardiographic results
        - **Max Heart Rate**: Maximum heart rate achieved
        - **Exercise Angina**: Exercise induced angina
        - **ST Depression**: ST depression induced by exercise
        """)
    
    with col2:
        # Target distribution pie chart
        target_counts = df['target'].value_counts()
        fig = px.pie(values=target_counts.values, 
                    names=['Healthy', 'Heart Disease'],
                    title="Heart Disease Distribution",
                    color_discrete_sequence=['#2ecc71', '#e74c3c'])
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

def show_analysis_page(df):
    """Display data analysis and visualizations"""
    st.markdown("## 📊 Data Analysis & Visualizations")
    
    # Age distribution
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(df, x='age', color='target', 
                          title='Age Distribution by Heart Disease Status',
                          labels={'target': 'Heart Disease', 'age': 'Age'},
                          color_discrete_map={0: '#2ecc71', 1: '#e74c3c'})
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Gender distribution
        gender_target = df.groupby(['sex', 'target']).size().reset_index(name='count')
        fig = px.bar(gender_target, x='sex', y='count', color='target',
                    title='Heart Disease by Gender',
                    labels={'target': 'Heart Disease', 'sex': 'Gender'},
                    color_discrete_map={0: '#2ecc71', 1: '#e74c3c'})
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Chest pain type analysis
    col1, col2 = st.columns(2)
    
    with col1:
        cp_target = df.groupby(['cp', 'target']).size().reset_index(name='count')
        fig = px.bar(cp_target, x='cp', y='count', color='target',
                    title='Heart Disease by Chest Pain Type',
                    labels={'target': 'Heart Disease', 'cp': 'Chest Pain Type'},
                    color_discrete_map={0: '#2ecc71', 1: '#e74c3c'})
        fig.update_layout(height=400, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Correlation heatmap
        numeric_cols = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'target']
        corr_data = df[numeric_cols].corr()
        
        fig = px.imshow(corr_data, 
                       title='Correlation Matrix of Numeric Features',
                       color_continuous_scale='RdBu_r',
                       aspect='auto')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Additional insights
    st.markdown("### Key Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        male_heart_disease = df[(df['sex'] == 'Male') & (df['target'] == 1)].shape[0]
        total_males = df[df['sex'] == 'Male'].shape[0]
        male_percentage = (male_heart_disease / total_males) * 100
        
        st.metric("Male Heart Disease Rate", f"{male_percentage:.1f}%")
    
    with col2:
        female_heart_disease = df[(df['sex'] == 'Female') & (df['target'] == 1)].shape[0]
        total_females = df[df['sex'] == 'Female'].shape[0]
        female_percentage = (female_heart_disease / total_females) * 100
        
        st.metric("Female Heart Disease Rate", f"{female_percentage:.1f}%")
    
    with col3:
        avg_age_heart_disease = df[df['target'] == 1]['age'].mean()
        st.metric("Avg Age with Heart Disease", f"{avg_age_heart_disease:.1f} years")

def show_prediction_page(model_df, le_dict):
    """Display prediction interface"""
    st.markdown("## 🤖 Heart Disease Prediction")
    
    st.markdown("### Enter Patient Information")
    
    # Create input form
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.slider("Age", 20, 80, 50)
        sex = st.selectbox("Sex", ["Male", "Female"])
        cp = st.selectbox("Chest Pain Type", 
                         ["typical angina", "atypical angina", "non-anginal", "asymptomatic"])
        trestbps = st.slider("Resting Blood Pressure", 90, 200, 120)
        chol = st.slider("Cholesterol", 100, 600, 200)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["FALSE", "TRUE"])
    
    with col2:
        restecg = st.selectbox("Resting ECG", 
                              ["normal", "st-t abnormality", "lv hypertrophy"])
        thalch = st.slider("Maximum Heart Rate", 60, 220, 150)
        exang = st.selectbox("Exercise Induced Angina", ["FALSE", "TRUE"])
        oldpeak = st.slider("ST Depression", 0.0, 6.0, 1.0, 0.1)
        slope = st.selectbox("Slope", ["upsloping", "flat", "downsloping"])
        ca = st.slider("Number of Major Vessels", 0, 3, 0)
        thal = st.selectbox("Thalassemia", ["normal", "fixed defect", "reversable defect"])
    
    if st.button("Predict Heart Disease", type="primary"):
        # Prepare input data
        input_data = {
            'age': age,
            'sex': sex,
            'cp': cp,
            'trestbps': trestbps,
            'chol': chol,
            'fbs': fbs,
            'restecg': restecg,
            'thalch': thalch,
            'exang': exang,
            'oldpeak': oldpeak,
            'slope': slope,
            'ca': ca,
            'thal': thal
        }
        
        # Encode categorical variables
        for col, value in input_data.items():
            if col in le_dict:
                try:
                    input_data[col] = le_dict[col].transform([str(value)])[0]
                except ValueError:
                    # Handle unseen labels
                    input_data[col] = 0
        
        # Prepare features for prediction
        X = model_df.drop('target', axis=1)
        y = model_df['target']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Train model
        model = LogisticRegression(random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Make prediction
        input_array = np.array(list(input_data.values())).reshape(1, -1)
        input_scaled = scaler.transform(input_array)
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]
        
        # Display result
        if prediction == 0:
            st.markdown(f"""
            <div class="prediction-result healthy">
                ✅ Low Risk of Heart Disease<br>
                Confidence: {probability[0]:.1%}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="prediction-result at-risk">
                ⚠️ High Risk of Heart Disease<br>
                Confidence: {probability[1]:.1%}
            </div>
            """, unsafe_allow_html=True)
        
        # Show probability breakdown
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Healthy Probability", f"{probability[0]:.1%}")
        with col2:
            st.metric("Heart Disease Probability", f"{probability[1]:.1%}")

def show_model_performance_page(model_df):
    """Display model performance metrics"""
    st.markdown("## 📈 Model Performance Analysis")
    
    # Prepare data
    X = model_df.drop('target', axis=1)
    y = model_df['target']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    results, trained_models = train_models(X_train_scaled, X_test_scaled, y_train, y_test)
    
    # Display results
    col1, col2 = st.columns(2)
    
    for i, (name, result) in enumerate(results.items()):
        with col1 if i % 2 == 0 else col2:
            st.markdown(f"### {name}")
            st.metric("Accuracy", f"{result['accuracy']:.3f}")
            
            # Confusion matrix
            cm = confusion_matrix(y_test, result['predictions'])
            fig = px.imshow(cm, 
                           title=f'{name} - Confusion Matrix',
                           labels=dict(x="Predicted", y="Actual"),
                           x=['Healthy', 'Heart Disease'],
                           y=['Healthy', 'Heart Disease'],
                           color_continuous_scale='Blues')
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance for Random Forest
    if 'Random Forest' in trained_models:
        st.markdown("### Feature Importance (Random Forest)")
        rf_model = trained_models['Random Forest']
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        fig = px.bar(feature_importance.head(10), 
                    x='importance', y='feature',
                    orientation='h',
                    title='Top 10 Most Important Features')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()