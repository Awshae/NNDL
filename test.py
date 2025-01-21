import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Load the dataset
data = pd.read_csv('city_day.csv')  # Replace with your dataset path

# Preprocess the data
# Drop rows where AQI is missing
data = data.dropna(subset=['AQI'])

# Define pollutant columns
pollutant_columns = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene']

# Fill missing pollutant values with column means
data[pollutant_columns] = data[pollutant_columns].fillna(data[pollutant_columns].mean())

# Encode the 'City' column
data['City'] = LabelEncoder().fit_transform(data['City'])
city_encoder = LabelEncoder()
data['City'] = city_encoder.fit_transform(data['City'])

# Convert 'Date' to datetime and extract year, month, and day
data['Date'] = pd.to_datetime(data['Date'])
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['Day'] = data['Date'].dt.day

# Drop unused columns
data = data.drop(columns=['Date', 'AQI_Bucket'])

# Features and target
X = data.drop(columns=['AQI'])  # Input features
y = data['AQI']  # Target variable

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Normalize the target (AQI)
y = (y - y.min()) / (y.max() - y.min())

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build a simple MLP model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1)  # Output layer for AQI
])
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=1)

# Evaluate the model
loss, mae = model.evaluate(X_test, y_test)
print(f"Test MAE: {mae}")

# User input for prediction
city_input = input("Enter the city name: ")
date_input = input("Enter the date (DD/MM/YYYY): ")

# Process user inputs
city_encoded = city_encoder.transform([city_input])[0]
date_parsed = pd.to_datetime(date_input, format='%d/%m/%Y')

year = date_parsed.year
month = date_parsed.month
day = date_parsed.day

# Create a sample input row (assuming mean values for pollutants)
input_data = [city_encoded] + [data[pollutant_columns].mean().tolist()] + [year, month, day]
input_data = scaler.transform([input_data])

# Predict AQI
predicted_aqi = model.predict(input_data)[0][0]
# Scale AQI back to original range
predicted_aqi = predicted_aqi * (y.max() - y.min()) + y.min()
print(f"Predicted AQI for {city_input} on {date_input}: {predicted_aqi}")
