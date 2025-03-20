import pandas as pd
import numpy as np

class BiasDetector:
    def __init__(self, dataset):
        self.dataset = dataset
        self.bias_metrics = {}
        
    def detect_bias(self):
        """Analyze dataset for various types of bias"""
        self.bias_metrics = {
            'statistical_parity': self.calculate_statistical_parity(),
            'disparate_impact': self.calculate_disparate_impact(),
            'equal_opportunity': self.calculate_equal_opportunity()
        }
        return self.bias_metrics
    
    def calculate_statistical_parity(self):
        """Calculate statistical parity difference"""
        sensitive_attrs = self.identify_sensitive_attributes()
        results = {}
        
        for attr in sensitive_attrs:
            groups = self.dataset.groupby(attr)
            group_proportions = groups.size() / len(self.dataset)
            results[attr] = group_proportions.std()
            
        return results
    
    def calculate_disparate_impact(self):
        """Calculate disparate impact ratio"""
        sensitive_attrs = self.identify_sensitive_attributes()
        results = {}
        
        for attr in sensitive_attrs:
            groups = self.dataset.groupby(attr)
            numeric_columns = self.dataset.select_dtypes(include=[np.number]).columns
            success_rates = groups[numeric_columns].mean()
            max_rate = success_rates.max()
            min_rate = success_rates.min()
            results[attr] = (min_rate / max_rate).min() if (max_rate > 0).any() else 1.0
            
        return results
    
    def calculate_equal_opportunity(self):
        """Calculate equal opportunity difference"""
        sensitive_attrs = self.identify_sensitive_attributes()
        target = self.identify_target_variable()
        results = {}
        
        if target is None:
            return results
            
        for attr in sensitive_attrs:
            groups = self.dataset.groupby(attr)
            tpr_diff = []
            
            for name, group in groups:
                true_positive_rate = self.calculate_tpr(group[target])
                tpr_diff.append(true_positive_rate)
                
            results[attr] = max(tpr_diff) - min(tpr_diff)
            
        return results
    
    def identify_sensitive_attributes(self):
        """Identify potential sensitive attributes in dataset"""
        categorical_cols = self.dataset.select_dtypes(
            include=['object', 'category']
        ).columns
        return [col for col in categorical_cols if self.dataset[col].nunique() < 10]
    
    def identify_target_variable(self):
        """Identify the target variable in the dataset"""
        binary_cols = [
            col for col in self.dataset.columns 
            if set(self.dataset[col].unique()) <= {0, 1}
        ]
        return binary_cols[0] if binary_cols else None
    
    def calculate_tpr(self, series):
        """Calculate True Positive Rate"""
        if len(series) == 0:
            return 0
        return (series == 1).sum() / len(series)