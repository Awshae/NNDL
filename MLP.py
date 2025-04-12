#import all required modules
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

#read the dataset
df = pd.read_csv("city_day.csv")

df.head()

#data preprocessing and EDA 
data_info = {
    "shape": df.shape,
    "columns": df.columns.tolist(),
    "head": df.head(),
    "info": df.info(),
    "missing_values": df.isnull().sum()
}

data_info

#dropping NaN values for AQI, filling others with median
df = df.dropna(subset="AQI")
df = df.drop(columns=["Date", "AQI_Bucket"], axis=1)
for col in df.columns:
    if df[col].isnull().sum()>0:
        df[col]=df[col].fillna(df[col].median())
df.isnull().sum()

#using LabelEncoder to use City: a str data column 
print(list(df['City'].unique())) #lists out all city names
label_encoder = LabelEncoder()
df['City'] = LabelEncoder().fit_transform(df['City']) 
print(list(df['City'].unique())) #shows after encoding

#correlation heatmap 
plt.figure(figsize=(12, 10))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap of Pollutants and AQI")
plt.show()

#training of the model
x = df.drop(columns=["AQI"])
y = df["AQI"]
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3, random_state=42)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

mlp = MLPRegressor(hidden_layer_sizes=(64, 32),
                   activation='logistic', solver='adam',
                    max_iter=1000, random_state=42)
mlp.fit(x_train, y_train)
y_pred = mlp.predict(x_test)

#evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error:", mae)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
r2 = r2_score(y_test, y_pred)
print("R-squared (R²) Score:", r2) 
plt.figure(figsize=(8, 6))

#result plotting
plt.scatter(y_test, y_pred, color='green', edgecolor='k', alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
plt.title("Scatter Plot: Predicted vs. Actual Performance Index", fontsize=14)
plt.xlabel("Actual Performance Index", fontsize=12)
plt.ylabel("Predicted Performance Index", fontsize=12)
plt.grid(alpha=0.3)
plt.show()

subset_size = 100 
points = np.arange(subset_size)
y_test_subset = y_test.values[:subset_size]
y_pred_subset = y_pred[:subset_size]
plt.figure(figsize=(12, 6))  # Adjust figure size
plt.plot(points, y_test_subset, color='red', label='Actual AQI')
plt.plot(points, y_pred_subset, color='green', label='Predicted AQI')
plt.title("Actual vs Predicted AQI (Subset)")
plt.xlabel("Data Points")
plt.ylabel("AQI")
plt.legend()
plt.grid(True)
plt.show()

