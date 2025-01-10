# Crop Recommendation System 🌾

## Overview
The **Crop Recommendation System** is a machine learning-powered application designed to assist farmers in selecting the most suitable crop based on environmental and soil conditions. This project combines data science and an interactive user interface to provide actionable insights, enabling smarter farming decisions.

---

## Features
- 🌱 **Crop Recommendation**: Predict the best crop based on soil nutrients, temperature, humidity, pH, and rainfall.
- 📊 **Data Insights**: Visualize feature correlations, crop distributions, and more.
- 📈 **Feature Importance**: Understand which factors influence crop recommendations the most.
- 💾 **Download Functionality**: Export recommendations as a CSV file.
- 🚀 **User-Friendly Interface**: Built with Streamlit for seamless interactivity.

---

## Technologies Used
- **Programming Language**: Python
- **Libraries**:
  - `pandas`, `numpy` (Data processing)
  - `scikit-learn` (Machine learning)
  - `matplotlib`, `seaborn` (Data visualization)
  - `streamlit` (Web application framework)

---

## Dataset
The project uses a crop dataset containing the following features:
- **N**: Nitrogen content in soil
- **P**: Phosphorus content in soil
- **K**: Potassium content in soil
- **Temperature**: Average environmental temperature (°C)
- **Humidity**: Percentage of moisture in the air
- **pH**: Acidity/alkalinity level of the soil
- **Rainfall**: Average rainfall (mm)

**Target**: Crop label (e.g., rice, wheat, maize).

---

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/nimramalik-4/crop-recommendation-system.git
   ```
2. Navigate to the project directory:
   ```bash
   cd crop-recommendation-system
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage
1. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
2. Input soil and environmental conditions in the sidebar.
3. Click the **"Recommend Crop"** button to get the best crop suggestion.
4. Explore data insights and feature importance using additional functionalities.

---

## Example
**Input:**  
- Nitrogen (N): 90, Phosphorus (P): 42, Potassium (K): 43
- Temperature: 23.5°C, Humidity: 82%, pH: 6.5, Rainfall: 200mm

**Output:**  
- **Recommended Crop**: Rice 🌾

## Future Enhancements
- Add support for more crop varieties and regions.
- Integrate additional environmental factors such as soil type.
- Develop a mobile-friendly version.

## Author
**Nimra**  
Passionate about data science, EDA, and impactful visualizations. Let's turn data into stories! 📊
