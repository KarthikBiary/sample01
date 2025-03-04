import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns

class SensorAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sensor Failure Prediction")
        self.root.geometry("1000x700")
        
        # Initialize variables
        self.df = None
        self.df_processed = None
        self.model = None
        
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.data_tab = ttk.Frame(self.notebook)
        self.model_tab = ttk.Frame(self.notebook)
        self.predict_tab = ttk.Frame(self.notebook)
        
        self.notebook.add(self.data_tab, text="Data")
        self.notebook.add(self.model_tab, text="Model")
        self.notebook.add(self.predict_tab, text="Predict")
        
        # Setup each tab
        self.setup_data_tab()
        self.setup_model_tab()
        self.setup_predict_tab()
    
    def setup_data_tab(self):
        load_frame = ttk.LabelFrame(self.data_tab, text="Load Data")
        load_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Add button to load CSV
        load_button = ttk.Button(load_frame, text="Load CSV File", command=self.load_csv)
        load_button.pack(side=tk.LEFT, padx=10, pady=10)
        
        sample_button = ttk.Button(load_frame, text="Use Sample Data", command=self.load_sample_data)
        sample_button.pack(side=tk.LEFT, padx=10, pady=10)
        
        self.status_label = ttk.Label(load_frame, text="No data loaded")
        self.status_label.pack(side=tk.LEFT, padx=10, pady=10)
        
        self.summary_frame = ttk.LabelFrame(self.data_tab, text="Data Summary")
        self.summary_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.summary_text = tk.Text(self.summary_frame, height=10)
        self.summary_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.viz_frame = ttk.LabelFrame(self.data_tab, text="Data Visualization")
        self.viz_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.feature_var = tk.StringVar()
        feature_label = ttk.Label(self.viz_frame, text="Select Feature:")
        feature_label.pack(side=tk.LEFT, padx=10, pady=10)
        self.feature_dropdown = ttk.Combobox(self.viz_frame, textvariable=self.feature_var, state="readonly")
        self.feature_dropdown.pack(side=tk.LEFT, padx=10, pady=10)
        self.feature_dropdown.bind("<<ComboboxSelected>>", self.update_visualization)
        
        plot_button = ttk.Button(self.viz_frame, text="Plot", command=self.update_visualization)
        plot_button.pack(side=tk.LEFT, padx=10, pady=10)
        
        self.fig_frame = ttk.Frame(self.data_tab)
        self.fig_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    def setup_model_tab(self):

        param_frame = ttk.LabelFrame(self.model_tab, text="Model Parameters")
        param_frame.pack(fill=tk.X, padx=10, pady=10)
        

        ttk.Label(param_frame, text="Number of Trees:").grid(row=0, column=0, padx=10, pady=10)
        self.n_estimators_var = tk.IntVar(value=100)
        ttk.Spinbox(param_frame, from_=10, to=500, textvariable=self.n_estimators_var, width=5).grid(row=0, column=1, padx=10, pady=10)
        
        ttk.Label(param_frame, text="Test Size (%):").grid(row=0, column=2, padx=10, pady=10)
        self.test_size_var = tk.IntVar(value=20)
        ttk.Spinbox(param_frame, from_=10, to=50, textvariable=self.test_size_var, width=5).grid(row=0, column=3, padx=10, pady=10)
        
        ttk.Label(param_frame, text="Random State:").grid(row=0, column=4, padx=10, pady=10)
        self.random_state_var = tk.IntVar(value=42)
        ttk.Spinbox(param_frame, from_=0, to=100, textvariable=self.random_state_var, width=5).grid(row=0, column=5, padx=10, pady=10)
        
        train_button = ttk.Button(param_frame, text="Train Model", command=self.train_model)
        train_button.grid(row=0, column=6, padx=10, pady=10)
        
        self.results_frame = ttk.LabelFrame(self.model_tab, text="Model Results")
        self.results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.results_text = tk.Text(self.results_frame, height=10)
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.importance_frame = ttk.LabelFrame(self.model_tab, text="Feature Importance")
        self.importance_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    def setup_predict_tab(self):
        # Create frame for prediction inputs
        input_frame = ttk.LabelFrame(self.predict_tab, text="Sensor Values")
        input_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # First row - battery voltage, engine temp, vibration
        ttk.Label(input_frame, text="Battery Voltage:").grid(row=0, column=0, padx=5, pady=5)
        self.battery_voltage_var = tk.DoubleVar(value=12.5)
        ttk.Entry(input_frame, textvariable=self.battery_voltage_var, width=8).grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(input_frame, text="Engine Temp:").grid(row=0, column=2, padx=5, pady=5)
        self.engine_temp_var = tk.DoubleVar(value=85.0)
        ttk.Entry(input_frame, textvariable=self.engine_temp_var, width=8).grid(row=0, column=3, padx=5, pady=5)
        
        ttk.Label(input_frame, text="Vibration:").grid(row=0, column=4, padx=5, pady=5)
        self.vibration_var = tk.DoubleVar(value=1.5)
        ttk.Entry(input_frame, textvariable=self.vibration_var, width=8).grid(row=0, column=5, padx=5, pady=5)
        
        # Second row - tire pressure, speed, brake wear
        ttk.Label(input_frame, text="Tire Pressure:").grid(row=1, column=0, padx=5, pady=5)
        self.tire_pressure_var = tk.DoubleVar(value=32.0)
        ttk.Entry(input_frame, textvariable=self.tire_pressure_var, width=8).grid(row=1, column=1, padx=5, pady=5)
        
        ttk.Label(input_frame, text="Speed:").grid(row=1, column=2, padx=5, pady=5)
        self.speed_var = tk.DoubleVar(value=45.0)
        ttk.Entry(input_frame, textvariable=self.speed_var, width=8).grid(row=1, column=3, padx=5, pady=5)
        
        ttk.Label(input_frame, text="Brake Wear:").grid(row=1, column=4, padx=5, pady=5)
        self.brake_wear_var = tk.DoubleVar(value=0.0)
        ttk.Entry(input_frame, textvariable=self.brake_wear_var, width=8).grid(row=1, column=5, padx=5, pady=5)
        
        # Third row - suspension, moving averages
        ttk.Label(input_frame, text="Suspension:").grid(row=2, column=0, padx=5, pady=5)
        self.suspension_var = tk.DoubleVar(value=0.0)
        ttk.Entry(input_frame, textvariable=self.suspension_var, width=8).grid(row=2, column=1, padx=5, pady=5)
        
        ttk.Label(input_frame, text="Battery MA:").grid(row=2, column=2, padx=5, pady=5)
        self.battery_ma_var = tk.DoubleVar(value=12.5)
        ttk.Entry(input_frame, textvariable=self.battery_ma_var, width=8).grid(row=2, column=3, padx=5, pady=5)
        
        ttk.Label(input_frame, text="Engine MA:").grid(row=2, column=4, padx=5, pady=5)
        self.engine_ma_var = tk.DoubleVar(value=85.0)
        ttk.Entry(input_frame, textvariable=self.engine_ma_var, width=8).grid(row=2, column=5, padx=5, pady=5)
        
        # Fourth row - vibration MA, hour, day of week, month
        ttk.Label(input_frame, text="Vibration MA:").grid(row=3, column=0, padx=5, pady=5)
        self.vibration_ma_var = tk.DoubleVar(value=1.5)
        ttk.Entry(input_frame, textvariable=self.vibration_ma_var, width=8).grid(row=3, column=1, padx=5, pady=5)
        
        ttk.Label(input_frame, text="Hour (0-23):").grid(row=3, column=2, padx=5, pady=5)
        self.hour_var = tk.IntVar(value=12)
        ttk.Spinbox(input_frame, from_=0, to=23, textvariable=self.hour_var, width=5).grid(row=3, column=3, padx=5, pady=5)
        
        ttk.Label(input_frame, text="Day (0-6):").grid(row=3, column=4, padx=5, pady=5)
        self.day_var = tk.IntVar(value=2)
        ttk.Spinbox(input_frame, from_=0, to=6, textvariable=self.day_var, width=5).grid(row=3, column=5, padx=5, pady=5)
        
        ttk.Label(input_frame, text="Month (1-12):").grid(row=4, column=0, padx=5, pady=5)
        self.month_var = tk.IntVar(value=6)
        ttk.Spinbox(input_frame, from_=1, to=12, textvariable=self.month_var, width=5).grid(row=4, column=1, padx=5, pady=5)
        
        predict_button = ttk.Button(input_frame, text="Predict", command=self.make_prediction)
        predict_button.grid(row=4, column=3, columnspan=2, padx=10, pady=10)
        
        self.prediction_frame = ttk.LabelFrame(self.predict_tab, text="Prediction Result")
        self.prediction_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.prediction_label = ttk.Label(self.prediction_frame, text="No prediction yet", font=("Arial", 14))
        self.prediction_label.pack(padx=10, pady=20)
        
        ttk.Label(self.prediction_frame, text="Failure Probability:").pack(padx=10, pady=5)
        self.probability_bar = ttk.Progressbar(self.prediction_frame, length=400, mode='determinate')
        self.probability_bar.pack(padx=10, pady=5)
        self.probability_label = ttk.Label(self.prediction_frame, text="0%")
        self.probability_label.pack(padx=10, pady=5)
    
    def load_csv(self):
        file_path = filedialog.askopenfilename(
            title="Select CSV file",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                self.df = pd.read_csv(file_path, parse_dates=['timestamp'])
                self.status_label.config(text=f"Loaded {len(self.df)} records")
                self.process_data()
                self.update_summary()
                self.update_feature_dropdown()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load CSV: {str(e)}")
    
    def load_sample_data(self):
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=1000, freq='H')
        
        battery_voltage = np.random.normal(12.5, 1.5, 1000)
        engine_temp = np.random.normal(85, 15, 1000)
        tire_pressure = np.random.normal(32, 5, 1000)
        vibration = np.random.normal(1.5, 1.0, 1000)
        speed = np.random.normal(45, 20, 1000)
        brake_wear = np.random.normal(0, 0.8, 1000)
        suspension_shocks = np.random.normal(0, 1.2, 1000)
        

        failure = np.zeros(1000, dtype=int)
        for i in range(1000):
            if (battery_voltage[i] < 10 and engine_temp[i] > 100) or vibration[i] > 3.0:
                failure[i] = 1
        
        self.df = pd.DataFrame({
            'timestamp': dates,
            'battery_voltage': battery_voltage,
            'engine_temp': engine_temp, 
            'tire_pressure': tire_pressure,
            'vibration': vibration,
            'speed': speed,
            'brake_wear': brake_wear,
            'suspension_shocks': suspension_shocks,
            'failure': failure
        })
        
        self.status_label.config(text=f"Loaded {len(self.df)} sample records")
        self.process_data()
        self.update_summary()
        self.update_feature_dropdown()
    
    def process_data(self):
        if self.df is None:
            return
        

        self.df_processed = self.df.copy()
        

        self.df_processed.dropna(inplace=True)
        

        self.df_processed['hour'] = self.df_processed['timestamp'].dt.hour
        self.df_processed['day_of_week'] = self.df_processed['timestamp'].dt.dayofweek
        self.df_processed['month'] = self.df_processed['timestamp'].dt.month
        

        window_size = 10
        self.df_processed['battery_voltage_ma'] = self.df_processed['battery_voltage'].rolling(window=window_size).mean()
        self.df_processed['engine_temp_ma'] = self.df_processed['engine_temp'].rolling(window=window_size).mean()
        self.df_processed['vibration_ma'] = self.df_processed['vibration'].rolling(window=window_size).mean()
        

        self.df_processed.dropna(inplace=True)
    
    def update_summary(self):
        if self.df_processed is None:
            return
        

        self.summary_text.delete(1.0, tk.END)
        

        summary = f"Dataset Summary:\n"
        summary += f"Total Records: {len(self.df_processed)}\n"
        summary += f"Failure Rate: {self.df_processed['failure'].mean() * 100:.2f}%\n\n"
        
        summary += "Column Statistics:\n"
        for col in self.df_processed.select_dtypes(include=[np.number]).columns:
            if col != 'timestamp':
                summary += f"{col}:\n"
                summary += f"  Min: {self.df_processed[col].min():.2f}\n"
                summary += f"  Max: {self.df_processed[col].max():.2f}\n"
                summary += f"  Mean: {self.df_processed[col].mean():.2f}\n"
                summary += f"  Std: {self.df_processed[col].std():.2f}\n\n"
        
        self.summary_text.insert(tk.END, summary)
    
    def update_feature_dropdown(self):
        if self.df_processed is None:
            return
        

        numeric_cols = self.df_processed.select_dtypes(include=[np.number]).columns.tolist()

        if 'failure' in numeric_cols:
            numeric_cols.remove('failure')
        

        self.feature_dropdown['values'] = numeric_cols
        if numeric_cols:
            self.feature_dropdown.current(0)
    
    def update_visualization(self, event=None):
        if self.df_processed is None or not self.feature_var.get():
            return
        

        for widget in self.fig_frame.winfo_children():
            widget.destroy()
        

        fig, ax = plt.subplots(figsize=(10, 5))
        

        feature = self.feature_var.get()
        

        sns.histplot(data=self.df_processed, x=feature, hue='failure', kde=True, ax=ax)
        ax.set_title(f'Distribution of {feature} by Failure Status')
        

        canvas = FigureCanvasTkAgg(fig, master=self.fig_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def train_model(self):
        if self.df_processed is None:
            messagebox.showwarning("Warning", "No data available for training.")
            return
        
        try:

            n_estimators = self.n_estimators_var.get()
            test_size = self.test_size_var.get() / 100  # Convert percentage to proportion
            random_state = self.random_state_var.get()

            X = self.df_processed.drop(['failure', 'timestamp'], axis=1, errors='ignore')
            y = self.df_processed['failure']
            

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            

            self.model = RandomForestClassifier(
                n_estimators=n_estimators,
                random_state=random_state
            )
            self.model.fit(X_train, y_train)
            

            y_pred = self.model.predict(X_test)
            

            accuracy = accuracy_score(y_test, y_pred)
            

            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, f"Model Training Results:\n\n")
            self.results_text.insert(tk.END, f"Accuracy: {accuracy:.4f}\n\n")
            self.results_text.insert(tk.END, "Classification Report:\n")
            self.results_text.insert(tk.END, classification_report(y_test, y_pred))
            self.results_text.insert(tk.END, "\nConfusion Matrix:\n")
            self.results_text.insert(tk.END, str(confusion_matrix(y_test, y_pred)))
            

            self.display_feature_importance(X.columns)
            
            messagebox.showinfo("Success", "Model trained successfully!")
            

            self.X_columns = X.columns
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to train model: {str(e)}")
    
    def display_feature_importance(self, feature_names):
        # Clear previous figure
        for widget in self.importance_frame.winfo_children():
            widget.destroy()

        fig, ax = plt.subplots(figsize=(8, 6))
        
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        

        sns.barplot(x=importances[indices], y=[feature_names[i] for i in indices], ax=ax)
        ax.set_title('Feature Importance')
        ax.set_xlabel('Importance')
        ax.set_ylabel('Feature')
        

        canvas = FigureCanvasTkAgg(fig, master=self.importance_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def make_prediction(self):
        if self.model is None:
            messagebox.showwarning("Warning", "No trained model available. Please train a model first.")
            return
        
        try:

            input_data = pd.DataFrame({
                'battery_voltage': [self.battery_voltage_var.get()],
                'engine_temp': [self.engine_temp_var.get()],
                'tire_pressure': [self.tire_pressure_var.get()],
                'vibration': [self.vibration_var.get()],
                'speed': [self.speed_var.get()],
                'brake_wear': [self.brake_wear_var.get()],
                'suspension_shocks': [self.suspension_var.get()],
                'hour': [self.hour_var.get()],
                'day_of_week': [self.day_var.get()],
                'month': [self.month_var.get()],
                'battery_voltage_ma': [self.battery_ma_var.get()],
                'engine_temp_ma': [self.engine_ma_var.get()],
                'vibration_ma': [self.vibration_ma_var.get()]
            })
            

            input_data = input_data[self.X_columns]

            prediction = self.model.predict(input_data)
            prediction_proba = self.model.predict_proba(input_data)

            if prediction[0] == 1:
                self.prediction_label.config(text="⚠️ FAILURE PREDICTED", foreground="red")
            else:
                self.prediction_label.config(text="✓ NO FAILURE PREDICTED", foreground="green")
            
            # Update probability bar
            probability = prediction_proba[0][1] * 100
            self.probability_bar['value'] = probability
            self.probability_label.config(text=f"{probability:.1f}%")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to make prediction: {str(e)}")


if __name__ == "__main__":
    root = tk.Tk()
    app = SensorAnalysisApp(root)
    root.mainloop()
