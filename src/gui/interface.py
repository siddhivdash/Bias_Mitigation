import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
from src.analysis.bias_detection import BiasDetector
from src.visualization.plots import DataVisualizer
from src.mitigation.bias_mitigation import BiasMitigator
from src.utils.data_processing import DataProcessor

class BiasDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Bias Detection Tool")
        self.root.geometry("1200x800")
        
        self.dataset = None
        self.mitigated_dataset = None
        
        self.setup_ui()
        
    def setup_ui(self):
        # Create main frames
        self.control_frame = ttk.Frame(self.root, padding="10")
        self.control_frame.pack(fill=tk.X)
        
        self.viz_frame = ttk.Frame(self.root, padding="10")
        self.viz_frame.pack(fill=tk.BOTH, expand=True)
        
        # Add controls
        self.upload_btn = ttk.Button(
            self.control_frame, 
            text="Upload Dataset",
            command=self.upload_dataset
        )
        self.upload_btn.pack(side=tk.LEFT, padx=5)
        
        self.analyze_btn = ttk.Button(
            self.control_frame,
            text="Analyze Bias",
            command=self.analyze_bias,
            state=tk.DISABLED
        )
        self.analyze_btn.pack(side=tk.LEFT, padx=5)
        
        self.mitigate_btn = ttk.Button(
            self.control_frame,
            text="Mitigate Bias",
            command=self.show_mitigation_dialog,
            state=tk.DISABLED
        )
        self.mitigate_btn.pack(side=tk.LEFT, padx=5)

    def upload_dataset(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("CSV Files", "*.csv")]
        )
        if file_path:
            try:
                self.dataset = pd.read_csv(file_path)
                self.analyze_btn.config(state=tk.NORMAL)
                messagebox.showinfo("Success", "Dataset loaded successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load dataset: {str(e)}")

    def analyze_bias(self):
        if self.dataset is not None:
            detector = BiasDetector(self.dataset)
            bias_results = detector.detect_bias()
            
            visualizer = DataVisualizer(self.viz_frame)
            visualizer.plot_bias_metrics(bias_results)
            
            self.mitigate_btn.config(state=tk.NORMAL)

    def show_mitigation_dialog(self):
        dialog = MitigationDialog(self.root, self)
        self.root.wait_window(dialog)

    def apply_mitigation(self, technique):
        mitigator = BiasMitigator(self.dataset)
        
        print(f"Applying mitigation technique: {technique}")
        print(f"Original dataset shape: {self.dataset.shape}")
        
        if technique == "resampling":
            self.mitigated_dataset = mitigator.apply_resampling()
        elif technique == "reweighting":
            self.mitigated_dataset = mitigator.apply_reweighting()
        elif technique == "synthetic":
            self.mitigated_dataset = mitigator.generate_synthetic_data()
        
        print(f"Mitigated dataset shape: {self.mitigated_dataset.shape}")
        
        visualizer = DataVisualizer(self.viz_frame)
        visualizer.plot_comparison(self.dataset, self.mitigated_dataset)
        
        # Save mitigated dataset
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv")]
        )
        if file_path:
            self.mitigated_dataset.to_csv(file_path, index=False)

class MitigationDialog(tk.Toplevel):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.title("Choose Bias Mitigation Technique")
        self.geometry("300x200")
        
        ttk.Label(
            self,
            text="Select mitigation technique:"
        ).pack(pady=10)
        
        techniques = ["resampling", "reweighting", "synthetic"]
        self.selected_technique = tk.StringVar(value=techniques[0])
        
        for technique in techniques:
            ttk.Radiobutton(
                self,
                text=technique.capitalize(),
                value=technique,
                variable=self.selected_technique
            ).pack(pady=5)
        
        ttk.Button(
            self,
            text="Apply",
            command=self.apply
        ).pack(pady=20)

    def apply(self):
        self.app.apply_mitigation(self.selected_technique.get())
        self.destroy()