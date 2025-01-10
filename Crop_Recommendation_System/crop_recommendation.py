# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

# Load Dataset
@st.cache_data
def load_data():
    # Replace with your dataset path
    data = pd.read_csv(r"C:/Users/CYBER-LAPTOP/Downloads/Crop_recommendation.csv")
    return data

data = load_data()

# Title & Introduction
st.title("🌾 **Enhanced Crop Recommendation System** 🌾")
st.markdown("""
This system helps farmers choose the best crops to grow based on soil and environmental conditions. 🌱  
Now with **enhanced features** for deeper insights and better interactivity! 🚀
""")

# Sidebar with Functionalities
st.sidebar.title("💡 **Explore Features** 💡")
st.sidebar.markdown("""
- 🌾 **Crop Recommendation**
- 📊 **Data Insights**
- 🔍 **Model Details**
- 📈 **Feature Importance**
""")

# 1️⃣ Dataset Preview
st.subheader("Dataset Overview 📊")
st.dataframe(data.head())

# 2️⃣ Data Insights
st.subheader("📊 **Data Insights**")
if st.checkbox("Show Data Distribution"):
    st.markdown("### Crop Distribution")
    crop_counts = data['label'].value_counts()
    st.bar_chart(crop_counts)

    st.markdown("### Feature Correlation Heatmap")
    # Filter numeric columns to avoid ValueError
    numeric_data = data.select_dtypes(include=['float64', 'int64'])  # Select only numeric columns
    fig, ax = plt.subplots()
    sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

# 3️⃣ Model Training & Feature Importance
X = data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

@st.cache_resource
def train_model():
    rf = RandomForestClassifier(random_state=42, n_estimators=100)
    rf.fit(X_train, y_train)
    return rf

model = train_model()

# Model Accuracy Display
st.sidebar.subheader("📈 **Model Accuracy**")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.sidebar.success(f"Accuracy: {accuracy * 100:.2f}%")

# Feature Importance
st.subheader("📈 **Feature Importance**")
importance = model.feature_importances_
feature_names = X.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

fig, ax = plt.subplots()
sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis', ax=ax)
st.pyplot(fig)

# 4️⃣ Interactive Input Section
st.subheader("🌾 **Enter Soil and Environmental Conditions** 🌾")
N = st.number_input("💧 Nitrogen Content (N)", min_value=0, max_value=200, step=1)
P = st.number_input("💧 Phosphorus Content (P)", min_value=0, max_value=200, step=1)
K = st.number_input("💧 Potassium Content (K)", min_value=0, max_value=200, step=1)
temperature = st.number_input("🌞 Temperature (°C)", min_value=-10.0, max_value=50.0, step=0.1)
humidity = st.number_input("🌦 Humidity (%)", min_value=0.0, max_value=100.0, step=0.1)
ph = st.number_input("💧 pH Level", min_value=0.0, max_value=14.0, step=0.1)
rainfall = st.number_input("🌧 Rainfall (mm)", min_value=0.0, max_value=500.0, step=0.1)

# Predict Button
if st.button("🚜 **Recommend Crop**"):
    input_features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    recommended_crop = model.predict(input_features)[0]
    st.success(f"🌱 **Recommended Crop**: **{recommended_crop}** 🎉")
    st.balloons()

# Download Recommendations
st.subheader("💾 **Download Your Recommendation**")
if st.button("📥 Download CSV"):
    recommendation_df = pd.DataFrame({
        'Nitrogen (N)': [N], 'Phosphorus (P)': [P], 'Potassium (K)': [K],
        'Temperature': [temperature], 'Humidity': [humidity],
        'pH': [ph], 'Rainfall': [rainfall], 'Recommended Crop': [recommended_crop]
    })
    csv = recommendation_df.to_csv(index=False).encode()
    st.download_button("Download CSV", csv, "crop_recommendation.csv", "text/csv", key='download-csv')
