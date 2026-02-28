# Heart Disease Prediction Dashboard

A comprehensive Streamlit web application for heart disease prediction using machine learning.

## Features

- **Interactive Dashboard**: Fully responsive design that works on desktop and mobile
- **Data Analysis**: Comprehensive visualizations and insights from the heart disease dataset
- **Real-time Predictions**: Input patient data and get instant heart disease risk predictions
- **Model Performance**: Compare different machine learning models and their accuracy
- **User-friendly Interface**: Clean, modern design with intuitive navigation

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Run the Streamlit app:
```bash
streamlit run app.py
```

3. Open your browser and navigate to `http://localhost:8501`

## Dataset

The application uses the UCI Heart Disease dataset which contains:
- 920 patient records
- 16 features including age, sex, chest pain type, blood pressure, cholesterol, etc.
- Binary target variable (0 = no heart disease, 1 = heart disease)

## Pages

### 🏠 Home
- Overview of the dataset
- Key statistics and metrics
- Heart disease distribution visualization

### 📊 Data Analysis
- Age and gender distribution analysis
- Chest pain type correlations
- Feature correlation heatmap
- Key insights and statistics

### 🤖 Prediction
- Interactive form for patient data input
- Real-time heart disease risk prediction
- Confidence scores and probability breakdown
- User-friendly result display

### 📈 Model Performance
- Comparison of multiple ML models (Logistic Regression, Random Forest)
- Accuracy metrics and confusion matrices
- Feature importance analysis
- Model evaluation visualizations

## Technologies Used

- **Streamlit**: Web application framework
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Plotly**: Interactive visualizations
- **Scikit-learn**: Machine learning models
- **Seaborn & Matplotlib**: Statistical visualizations

## Responsive Design

The application is fully responsive and includes:
- Mobile-friendly layout
- Adaptive column layouts
- Responsive charts and visualizations
- Touch-friendly interface elements
- Custom CSS for optimal viewing on all devices