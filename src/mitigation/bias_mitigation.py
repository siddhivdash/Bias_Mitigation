import pandas as pd
import numpy as np
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE

class BiasMitigator:
    def __init__(self, dataset):
        self.dataset = dataset
        
    def apply_resampling(self):
        """Apply resampling technique to balance the dataset"""
        sensitive_attrs = self._identify_sensitive_attributes()
        balanced_data = self.dataset.copy()
        
        for attr in sensitive_attrs:
            # Get the size of the largest group
            group_sizes = balanced_data[attr].value_counts()
            max_size = group_sizes.max()
            
            # Resample each group to match the largest group
            resampled_groups = []
            for group_val in group_sizes.index:
                group_data = balanced_data[balanced_data[attr] == group_val]
                if len(group_data) < max_size:
                    resampled_group = resample(
                        group_data,
                        replace=True,
                        n_samples=max_size,
                        random_state=42
                    )
                    resampled_groups.append(resampled_group)
                else:
                    resampled_groups.append(group_data)
            
            balanced_data = pd.concat(resampled_groups)
        
        print("Resampling applied:")
        print(balanced_data[sensitive_attrs].value_counts())
        return balanced_data
    
    def apply_reweighting(self):
        """Apply reweighting technique to balance the dataset"""
        sensitive_attrs = self._identify_sensitive_attributes()
        weighted_data = self.dataset.copy()
        
        for attr in sensitive_attrs:
            # Calculate group frequencies
            group_freqs = weighted_data[attr].value_counts(normalize=True)
            
            # Calculate weights (inverse of frequency)
            weights = 1 / group_freqs
            
            # Normalize weights
            weights = weights / weights.sum()
            
            # Apply weights to each group
            weighted_data['weight'] = weighted_data[attr].map(weights)
        
        # Duplicate rows based on weights
        weighted_data = weighted_data.loc[weighted_data.index.repeat(weighted_data['weight'].round().astype(int))]
        weighted_data = weighted_data.drop(columns=['weight'])
        
        print("Reweighting applied:")
        print(weighted_data[sensitive_attrs].value_counts())
        return weighted_data
    
    def generate_synthetic_data(self):
        """Generate synthetic data using SMOTE"""
        sensitive_attrs = self._identify_sensitive_attributes()
        target = self._identify_target_variable()
        
        if target is None:
            return self.dataset.copy()
            
        # Prepare data for SMOTE
        X = self.dataset.drop(columns=[target])
        y = self.dataset[target]
        
        # Convert categorical variables to numeric
        X_encoded = pd.get_dummies(X)
        
        # Apply SMOTE
        smote = SMOTE(random_state=42)
        X_synthetic, y_synthetic = smote.fit_resample(X_encoded, y)
        
        # Convert back to original format
        synthetic_data = self._reconstruct_categorical_data(
            X_synthetic,
            X_encoded.columns,
            sensitive_attrs
        )
        synthetic_data[target] = y_synthetic
        
        print("Synthetic data generation applied:")
        print(synthetic_data[sensitive_attrs].value_counts())
        return synthetic_data
    
    def _identify_sensitive_attributes(self):
        """Identify potential sensitive attributes in dataset"""
        categorical_cols = self.dataset.select_dtypes(
            include=['object', 'category']
        ).columns
        return [col for col in categorical_cols if self.dataset[col].nunique() < 10]
    
    def _identify_target_variable(self):
        """Identify the target variable in the dataset"""
        binary_cols = [
            col for col in self.dataset.columns 
            if set(self.dataset[col].unique()) <= {0, 1}
        ]
        return binary_cols[0] if binary_cols else None
    
    def _reconstruct_categorical_data(self, X_synthetic, columns, sensitive_attrs):
        """Reconstruct categorical variables from one-hot encoded data"""
        reconstructed = pd.DataFrame(X_synthetic, columns=columns)
        
        for attr in sensitive_attrs:
            # Get columns that were one-hot encoded from this attribute
            attr_cols = [col for col in columns if col.startswith(f"{attr}_")]
            
            # Convert back to categorical
            max_cols = reconstructed[attr_cols].idxmax(axis=1)
            reconstructed[attr] = max_cols.apply(lambda x: x.replace(f"{attr}_", ""))
            
            # Drop the one-hot encoded columns
            reconstructed = reconstructed.drop(columns=attr_cols)
            
        return reconstructed