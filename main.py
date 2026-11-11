import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
import numpy as np 
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import os

# Try to import customtkinter for better UI
try:
    import customtkinter as ctk
    CTK_AVAILABLE = True
except ImportError:
    CTK_AVAILABLE = False
    print("Install customtkinter for better UI: pip install customtkinter")

# ----------------------------
# ML Model for Disease Prediction
# ----------------------------
class DiseasePredictor:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.train_dummy_model()

    def train_dummy_model(self):
        """Generates synthetic training data and trains the Random Forest model."""
        np.random.seed(42)
        # Features: Fever, Chills, Headache, Excessive Hunger, Excessive Urination, Yellow Skin, Abdominal Pain, Joint Pain, Family History
        X = np.random.randint(0, 4, size=(500, 9))
        # Target: Diseases (0: Malaria, 1: Hepatitis, 2: Diabetes, 3: Healthy)
        y = np.random.randint(0, 4, size=(500,))
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True

    def predict(self, symptoms):
        """Predicts probabilities for diseases based on input symptoms."""
        if not self.is_trained:
            return None
        input_data = np.array([symptoms])
        input_scaled = self.scaler.transform(input_data)
        probabilities = self.model.predict_proba(input_scaled)[0]
        return probabilities

# ----------------------------
# Main Application GUI
# ----------------------------
class HealthApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Health Monitoring System")
        self.root.geometry("1000x700")
        
        self.predictor = DiseasePredictor()
        
        # UI Styling / Theme setup
        if CTK_AVAILABLE:
            ctk.set_appearance_mode("Dark")
            ctk.set_default_color_theme("blue")
        
        self.create_widgets()

    def create_widgets(self):
        # Main Container Frame
        self.main_frame = ttk.Frame(self.root, padding=10)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Title Label
        title_label = ttk.Label(self.main_frame, text="AI Health Monitor & Disease Predictor", font=("Arial", 16, "bold"))
        title_label.pack(pady=10)

        # Content Layout (Left: Inputs, Right: Visualizations)
        content_frame = ttk.Frame(self.main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # Left Frame - Symptom Inputs
        input_frame = ttk.LabelFrame(content_frame, text=" Enter Symptoms (0: None, 1: Mild, 2: Moderate, 3: Severe) ", padding=10)
        input_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        self.symptom_vars = {}
        symptoms_list = [
            "Fever", "Chills", "Headache", "Excessive Hunger", 
            "Excessive Urination", "Yellow Skin", "Abdominal Pain", 
            "Joint Pain", "Family History"
        ]

        for idx, symptom in enumerate(symptoms_list):
            lbl = ttk.Label(input_frame, text=symptom, font=("Arial", 10))
            lbl.grid(row=idx, column=0, sticky=tk.W, pady=5)
            
            var = tk.IntVar(value=0)
            combo = ttk.Combobox(input_frame, textvariable=var, values=[0, 1, 2, 3], width=10, state="readonly")
            combo.grid(row=idx, column=1, padx=10, pady=5)
            self.symptom_vars[symptom] = var

        # Action Buttons
        btn_frame = ttk.Frame(input_frame)
        btn_frame.grid(row=len(symptoms_list), column=0, columnspan=2, pady=15)

        analyze_btn = ttk.Button(btn_frame, text="Analyze Health", command=self.analyze_health)
        analyze_btn.pack(side=tk.LEFT, padx=5)

        pdf_btn = ttk.Button(btn_frame, text="Generate PDF Report", command=self.generate_pdf_report)
        pdf_btn.pack(side=tk.LEFT, padx=5)

        # Health Score Display
        self.score_label = ttk.Label(input_frame, text="Overall Health Score: --", font=("Arial", 12, "bold"))
        self.score_label.grid(row=len(symptoms_list)+1, column=0, columnspan=2, pady=10)

        # Right Frame - Matplotlib Graphs Dashboard
        self.graph_frame = ttk.LabelFrame(content_frame, text=" Health Analysis Dashboard ", padding=10)
        self.graph_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)

        self.fig, self.ax = plt.subplots(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.graph_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.plot_initial_chart()

    def plot_initial_chart(self):
        self.ax.clear()
        diseases = ['Malaria', 'Hepatitis', 'Diabetes', 'Healthy']
        probs = [0.0, 0.0, 0.0, 100.0]
        self.ax.bar(diseases, probs, color=['#ff9999','#66b3ff','#99ff99','#ffcc99'])
        self.ax.set_ylim(0, 100)
        self.ax.set_ylabel("Probability (%)")
        self.ax.set_title("Disease Probability Distribution")
        self.canvas.draw()

    def analyze_health(self):
        symptoms_values = [var.get() for var in self.symptom_vars.values()]
        probs = self.predictor.predict(symptoms_values)
        if probs is None:
            messagebox.showerror("Error", "Model is not trained yet!")
            return

        total_severity = sum(symptoms_values)
        health_score = max(0.0, 100.0 - (total_severity * 3.5))
        
        self.score_label.config(text=f"Overall Health Score: {health_score:.1f}%")

        self.ax.clear()
        diseases = ['Malaria', 'Hepatitis', 'Diabetes', 'Healthy']
        percentages = [p * 100 for p in probs]
        
        self.ax.bar(diseases, percentages, color=['#e74c3c', '#3498db', '#2ecc71', '#f1c40f'])
        self.ax.set_ylim(0, 100)
        self.ax.set_ylabel("Probability (%)")
        self.ax.set_title("Disease Probability (%)")
        self.canvas.draw()

    def generate_pdf_report(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF files", "*.pdf")])
        if not file_path:
            return
        
        try:
            with PdfPages(file_path) as pdf:
                fig_pdf, ax_pdf = plt.subplots(figsize=(8, 6))
                ax_pdf.axis('off')
                ax_pdf.text(0.1, 0.8, "HEALTH MONITORING REPORT", fontsize=18, fontweight='bold')
                ax_pdf.text(0.1, 0.7, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", fontsize=12)
                ax_pdf.text(0.1, 0.6, self.score_label.cget("text"), fontsize=14, fontweight='bold')
                
                y_pos = 0.45
                ax_pdf.text(0.1, 0.5, "SYMPTOMS ENTERED:", fontsize=12, fontweight='bold')
                for symptom, var in self.symptom_vars.items():
                    ax_pdf.text(0.1, y_pos, f"- {symptom}: {var.get()}", fontsize=10)
                    y_pos -= 0.04

                pdf.savefig(fig_pdf)
                plt.close(fig_pdf)

            messagebox.showinfo("Success", f"PDF Report saved successfully at:\n{file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate PDF: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = HealthApp(root)
    root.mainloop()
