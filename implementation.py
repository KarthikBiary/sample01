import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('/content/buggy_sensor_data.csv', parse_dates=['timestamp'])


print("Missing values:\n", df.isnull().sum())

df.dropna(inplace=True)


df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['month'] = df['timestamp'].dt.month

df.drop('timestamp', axis=1, inplace=True)


window_size = 10
df['battery_voltage_ma'] = df['battery_voltage'].rolling(window=window_size).mean()
df['engine_temp_ma'] = df['engine_temp'].rolling(window=window_size).mean()
df['vibration_ma'] = df['vibration'].rolling(window=window_size).mean()

df.dropna(inplace=True)

X = df.drop('failure', axis=1)
y = df['failure']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))


feature_importances = model.feature_importances_
features = X.columns
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances, y=features)
plt.title('Feature Importance')
plt.show()


new_data = pd.DataFrame({
    'battery_voltage': [10.5],
    'engine_temp': [17],
    'tire_pressure': [27],
    'vibration': [2],
    'speed': [5],
    'brake_wear': [-2],
    'suspension_shocks': [1],
    'hour': [10],
    'day_of_week': [2],
    'month': [10],
    'battery_voltage_ma': [12.4],
    'engine_temp_ma': [84],
    'vibration_ma': [2.1]
}, index=[0])

new_data = new_data[X_train.columns]

prediction = model.predict(new_data)
print("Prediction for new data:", "Failure" if prediction[0] == 1 else "No Failure")