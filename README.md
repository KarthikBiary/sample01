    Realistic Patterns:

        Gradual battery degradation with daily recharge cycles

        Engine temperature correlated with speed

        Progressive brake wear

        Slow tire pressure leaks

    Anomalies & Failures:

        Sudden battery voltage drops

        Engine overheating events

        Random vibration spikes

        Monthly tire pressure anomalies

    Maintenance Events:

        Simulated battery replacements

        Daily tire inflation

    Failure Labels:

        Binary failure column indicating abnormal conditions

Dataset Columns:

    timestamp: DateTime index

    battery_voltage: Simulated battery state (Volts)

    engine_temp: Engine temperature (°C)

    tire_pressure: Tire pressure (PSI)

    vibration: Vibration levels (arbitrary units)

    speed: Vehicle speed (km/h)

    brake_wear: Brake pad wear percentage

    suspension_shocks: Shock absorption events

    failure: Binary failure indicator (0 = normal, 1 = failure)

Explanation of the Code
1. Preprocess the Data

    Handle Missing Values: Drop rows with missing data.

    Time-Based Features: Extract hour, day_of_week, and month from the timestamp.

    Drop Timestamp: Remove the timestamp column as it's not needed for modeling.

2. Feature Engineering

    Rolling Window Features: Add moving averages for key parameters (battery_voltage, engine_temp, vibration) to capture trends.

    Separate Features and Target: Split the dataset into features (X) and the target variable (y).

3. Train a Machine Learning Model

    Random Forest Classifier: A robust algorithm for classification tasks.

    Train-Test Split: Split the data into training (80%) and testing (20%) sets.

4. Evaluate the Model

    Classification Report: Precision, recall, F1-score, and support.

    Confusion Matrix: Visualize true positives, false positives, etc.

    Feature Importance: Identify which features contribute most to the model's predictions.

5. Deploy for Real-Time Predictions

    New Data Prediction: Simulate real-time predictions using a new data point.
   
Steps to Build the GUI

    Create a Tkinter Window: Design the layout with buttons, labels, and text boxes.

    Add File Upload Functionality: Allow users to upload a CSV file.

    Run the Predictive Algorithm: Process the uploaded data and display the results.

    Display Results: Show the classification report, confusion matrix, and feature importance plot.
