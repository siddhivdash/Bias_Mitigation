import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
import numpy as np
from src.analysis.bias_detection import BiasDetector

class DataVisualizer:
    def __init__(self, master):
        self.master = master
        self.figures = []
        
    def clear_plots(self):
        """Clear existing plots"""
        for widget in self.master.winfo_children():
            widget.destroy()
        self.figures = []
        
    def plot_bias_metrics(self, bias_results):
        """Create visualizations for bias metrics"""
        self.clear_plots()
        
        # Create figure for statistical parity
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        self._plot_statistical_parity(ax1, bias_results['statistical_parity'])
        self.add_plot_to_gui(fig1, 0, 0)
        
        # Create figure for disparate impact
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        self._plot_disparate_impact(ax2, bias_results['disparate_impact'])
        self.add_plot_to_gui(fig2, 0, 1)
        
        # Create figure for equal opportunity
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        self._plot_equal_opportunity(ax3, bias_results['equal_opportunity'])
        self.add_plot_to_gui(fig3, 1, 0)
        
    def plot_comparison(self, original_data, mitigated_data):
        """Create comparison visualizations"""
        self.clear_plots()
        
        # Distribution comparison
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        self._plot_distribution_comparison(ax1, original_data, mitigated_data)
        self.add_plot_to_gui(fig1, 0, 0)
        
        # Bias metrics comparison
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        self._plot_metrics_comparison(ax2, original_data, mitigated_data)
        self.add_plot_to_gui(fig2, 0, 1)
        
    def add_plot_to_gui(self, figure, row, col):
        """Add a matplotlib figure to the GUI"""
        canvas = FigureCanvasTkAgg(figure, master=self.master)
        canvas.draw()
        canvas.get_tk_widget().grid(row=row, column=col, padx=5, pady=5)
        self.figures.append(figure)
        
    def _plot_statistical_parity(self, ax, stats):
        if not stats:
            ax.text(0.5, 0.5, 'No Data Available', horizontalalignment='center', verticalalignment='center')
            return
        
        attributes = list(stats.keys())
        values = list(stats.values())
        
        ax.bar(attributes, values)
        ax.set_title('Statistical Parity Difference', pad=20)
        ax.set_ylabel('Difference')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
    def _plot_disparate_impact(self, ax, stats):
        if not stats:
            ax.text(0.5, 0.5, 'No Data Available', horizontalalignment='center', verticalalignment='center')
            return
        
        attributes = list(stats.keys())
        values = list(stats.values())
        
        ax.bar(attributes, values)
        ax.axhline(y=0.8, color='r', linestyle='--', label='0.8 threshold')
        ax.set_title('Disparate Impact Ratio', pad=20)
        ax.set_ylabel('Ratio')
        plt.xticks(rotation=45, ha='right')
        ax.legend()
        plt.tight_layout()
        
    def _plot_equal_opportunity(self, ax, stats):
        if not stats:
            ax.text(0.5, 0.5, 'No Data Available', horizontalalignment='center', verticalalignment='center')
            return
        
        attributes = list(stats.keys())
        values = list(stats.values())
        
        ax.bar(attributes, values)
        ax.set_title('Equal Opportunity Difference', pad=20)
        ax.set_ylabel('Difference')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
    def _plot_distribution_comparison(self, ax, original, mitigated):
        # Example: Plot distribution of a sensitive attribute
        sensitive_attr = original.select_dtypes(include=['object']).columns[0]
        
        orig_dist = original[sensitive_attr].value_counts(normalize=True)
        mit_dist = mitigated[sensitive_attr].value_counts(normalize=True)
        
        # Reindex to ensure both distributions have the same categories
        all_categories = orig_dist.index.union(mit_dist.index)
        orig_dist = orig_dist.reindex(all_categories, fill_value=0)
        mit_dist = mit_dist.reindex(all_categories, fill_value=0)
        
        x = np.arange(len(orig_dist))
        width = 0.35
        
        ax.bar(x - width/2, orig_dist, width, label='Original')
        ax.bar(x + width/2, mit_dist, width, label='Mitigated')
        
        ax.set_title(f'Distribution Comparison - {sensitive_attr}', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(orig_dist.index, rotation=45, ha='right')
        ax.legend()
        plt.tight_layout()
        
    def _plot_metrics_comparison(self, ax, original, mitigated):
        # Example: Compare bias metrics before and after mitigation
        detector_orig = BiasDetector(original)
        detector_mit = BiasDetector(mitigated)
        
        metrics_orig = detector_orig.detect_bias()
        metrics_mit = detector_mit.detect_bias()
        
        # Plot comparison of statistical parity
        stats_orig = metrics_orig['statistical_parity']
        stats_mit = metrics_mit['statistical_parity']
        
        attributes = list(stats_orig.keys())
        orig_values = list(stats_orig.values())
        mit_values = list(stats_mit.values())
        
        x = np.arange(len(attributes))
        width = 0.35
        
        ax.bar(x - width/2, orig_values, width, label='Original')
        ax.bar(x + width/2, mit_values, width, label='Mitigated')
        
        ax.set_title('Bias Metrics Comparison', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(attributes, rotation=45, ha='right')
        ax.legend()
        plt.tight_layout()