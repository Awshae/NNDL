import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
data = pd.read_csv('city_day.csv')
data = data.dropna(subset=['AQI'])

# Define pollutant columns
pollutant_columns = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene']

# Fill missing pollutant values with column means
data[pollutant_columns] = data[pollutant_columns].fillna(data[pollutant_columns].mean())

# Encode the 'City' column
city_encoder = LabelEncoder()
data['City'] = city_encoder.fit_transform(data['City'])

# Convert 'Date' to datetime and extract features
data['Date'] = pd.to_datetime(data['Date'])
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['Day'] = data['Date'].dt.day

# Drop unused columns
data = data.drop(columns=['Date', 'AQI_Bucket'])

# Features and target
X = data.drop(columns=['AQI'])  # Input features
y = data['AQI']  # Target variable

# Normalize features using MinMaxScaler
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Use Kernel Ridge Regression with RBF Kernel
kr = KernelRidge(kernel='rbf', alpha=1.0, gamma=0.1)
kr.fit(X_train, y_train)

# Evaluate the model
y_pred = kr.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Test MAE: {mae}")
print(f"R-squared (R²): {r2}")

# User input for prediction
print("---------------Prediction Model for the following cities---------------\nAhmedabad,Aizawl,Amaravati,Amritsar,Bengaluru,\nBhopal,Brajrajnagar,Chandigarh,Chennai,Coimbatore,\nDelhi,Ernakulam,Gurugram,Guwahati,Hyderabad,\nJaipur,Jorapokhar,Kochi,Kolkata,Lucknow,\nMumbai,Patna,Shillong,Talcher,Thiruvananthapuram and Visakhapatnam")

city_input = input("Enter the city name: ")
date_input = input("Enter the date (DD/MM/YYYY): ")

# Process user inputs
city_encoded = city_encoder.transform([city_input])[0]
date_parsed = pd.to_datetime(date_input, format='%d/%m/%Y')
year, month, day = date_parsed.year, date_parsed.month, date_parsed.day

# Create a sample input row (assuming mean values for pollutants)
input_data = pd.DataFrame([[city_encoded] + list(data[pollutant_columns].mean()) + [year, month, day]], 
                          columns=data.drop(columns=['AQI']).columns)

# Apply transformations
input_data = scaler.transform(input_data)

# Predict AQI
predicted_aqi = kr.predict(input_data)[0]
print(f"Predicted AQI for {city_input} on {date_input}: {predicted_aqi}")

# Correlation Heatmap
plt.figure(figsize=(15, 9))
correlation = data.corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation between Features and AQI")
plt.show()
