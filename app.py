import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from PIL import Image
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Set page title and layout
st.set_page_config(page_title="Housing Price Prediction", layout="wide")

# Load the data
data = pd.read_csv('Data/Housing.csv')

# Drop the 'prefarea' and 'stories' columns from the data
data.drop(['prefarea', 'stories'], axis=1, inplace=True)

# Reorder columns to make sure 'price' is the last column
data = data[[col for col in data.columns if col != 'price'] + ['price']]

# Encode categorical variables
binary_columns = ["mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning"]
for col in binary_columns:
    data[col] = data[col].map({"yes": 1, "no": 0})

# Encode 'furnishingstatus'
furnishing_mapping = {"unfurnished": 0, "semi-furnished": 1, "furnished": 2}
data["furnishingstatus"] = data["furnishingstatus"].map(furnishing_mapping)

# Load the model and the scaler from the saved file
with open('model/linear_regression_model.pkl', 'rb') as file:
    model = pickle.load(file)

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(data.drop(columns=["price"]))  
y_scaled = scaler_y.fit_transform(data["price"].values.reshape(-1, 1))

# Prepare the features
X = data.drop(columns=['price'])
numerical_columns = X.select_dtypes(include=['int64', 'float64']).columns

# Home 
st.title('House Price Prediction')
# image = Image.open("media/Background.jpg")
# small_image = image.resize((800, 400))
# st.image(small_image, use_column_width=False)

# Navigation menu for main content
page = st.radio("Select a page", ["Data", "Visualization", "House Price Prediction (Model)"], index=0)

# Page 1: Data
if page == "Data":
    st.title('Housing Data')
    st.write("Here is the dataset used for training the model.")
    st.dataframe(data)

# Page 2: Visualization
elif page == "Visualization":
    st.title('Data Visualization')

    # 1. Correlation Heatmap
    st.subheader("Correlation Between Data")
    corr_matrix = data.corr()  
    plt.figure(figsize=(10, 8))  
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm") 
    st.pyplot(plt) 

    # 2. Histograms for 'area', 'bedrooms', 'parking' vs 'price'
    st.subheader("Histograms for All Columns")
    data.hist(figsize=(10, 10), bins=10)
    plt.suptitle("Histograms for All Columns", fontsize=16)
    st.pyplot(plt)

    # 3. Scatter Plot: Price vs Area
    st.subheader("Price vs Area")
    x = data['area']
    y = data['price']
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, color='blue', alpha=0.8, edgecolors='k')
    plt.xlabel("Area")
    plt.ylabel("Price")
    plt.title("Price vs Area")
    plt.grid(True)
    st.pyplot(plt)

    # 4. Scatter Plot: Predicted vs Target Values
    st.subheader("Linear Regression Predicted vs Target Values")
    st.image("media/model.png", use_column_width=True)

# Page 3: Price Prediction
elif page == "House Price Prediction (Model)":
    st.title("House Price Prediction")

    # Get user inputs
    area = st.number_input("Area (sq ft)", min_value=0, max_value=10000, step=10)
    bedrooms = st.number_input("Number of Bedrooms", min_value=1, max_value=10, step=1)
    bathrooms = st.number_input("Number of Bathrooms", min_value=1, max_value=10, step=1)
    mainroad = st.selectbox("Main Road Access", ["yes", "no"])
    guestroom = st.selectbox("Guest Room Available", ["yes", "no"])
    basement = st.selectbox("Has Basement", ["yes", "no"])
    hotwaterheating = st.selectbox("Hot Water Heating", ["yes", "no"])
    airconditioning = st.selectbox("Air Conditioning", ["yes", "no"])
    parking = st.number_input("Number of Parking Spaces", min_value=0, max_value=5, step=1)
    furnishingstatus = st.selectbox("Furnishing Status", ["furnished", "semi-furnished", "unfurnished"])

    # Convert categorical values to numeric
    mainroad = 1 if mainroad == "yes" else 0
    guestroom = 1 if guestroom == "yes" else 0
    basement = 1 if basement == "yes" else 0
    hotwaterheating = 1 if hotwaterheating == "yes" else 0
    airconditioning = 1 if airconditioning == "yes" else 0

    furnishing_mapping = {"unfurnished": 0, "semi-furnished": 1, "furnished": 2}
    furnishingstatus = furnishing_mapping[furnishingstatus]

    # Prepare input data
    input_data = np.array([[area, bedrooms, bathrooms, mainroad, guestroom, basement,
                            hotwaterheating, airconditioning, parking, furnishingstatus]])

    # Scale the input data
    input_data_scaled = scaler_X.transform(input_data)

    # Predict the price
    if st.button("Predict Price"):
        prediction_scaled = model.predict(input_data_scaled)  # Predict the scaled price
        predicted_price = scaler_y.inverse_transform([[prediction_scaled[0]]])[0, 0]  # Inverse transform to get original price

        st.success(f"Predicted House Price: ${predicted_price:,.2f}")
