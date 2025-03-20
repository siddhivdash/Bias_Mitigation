import pandas as pd
import numpy as np
from sklearn.utils import resample
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from imblearn.over_sampling import ADASYN


class BiasMitigator:
    def __init__(self, dataset):
        self.dataset = dataset
        
    def apply_reweighting(self):
        sensitive_attrs = self._identify_sensitive_attributes()
        weighted_data = self.dataset.copy()
        n_rows = len(weighted_data)
        
        if 'weight' in weighted_data.columns:
            weighted_data = weighted_data.drop(columns=['weight'])
        
        if not sensitive_attrs:
            weighted_data['weight'] = 1.0
            print("No sensitive attributes provided. Assigned uniform weights.")
        else:
            weighted_data['temp_group'] = weighted_data[sensitive_attrs].agg('-'.join, axis=1)
            group_counts = weighted_data['temp_group'].value_counts()
            group_freqs = group_counts / n_rows
            weights = 1 / group_freqs
            weights = weights * n_rows / weights.sum()
            weighted_data['weight'] = weighted_data['temp_group'].map(weights)
            weighted_data = weighted_data.drop(columns=['temp_group'])
        
        total_weight = weighted_data['weight'].sum()
        print("Reweighting applied:")
        print(f"Total sum of weights: {total_weight:.2f}")
        if abs(total_weight - n_rows) > 1e-6:
            print(f"Error: Weight sum does not match expected value! Expected: {n_rows}, Got: {total_weight}")
            raise ValueError("Weight calculation failed!")
        
        for attr in sensitive_attrs:
            print(f"\nBalance for {attr}:")
            attr_sums = weighted_data.groupby(attr)['weight'].sum()
            print("Actual sums:")
            print(attr_sums)
            print("Normalized:")
            print(attr_sums / attr_sums.sum())
        
        return weighted_data
    
    def apply_resampling(self):
        """Apply resampling technique to balance the dataset"""
        sensitive_attrs = self._identify_sensitive_attributes()
        balanced_data = self.dataset.copy()
        original_size = len(balanced_data)

        # Create a combined group identifier based on all sensitive attributes
        if sensitive_attrs:
            # Create a temporary column with combined group values
            balanced_data['temp_group'] = balanced_data[sensitive_attrs].agg('-'.join, axis=1)
            
            # Get group sizes
            group_sizes = balanced_data['temp_group'].value_counts()
            n_groups = len(group_sizes)
            
            # Set target size: aim to keep total size close to original, but at least 100 per group
            target_size = max(100, min(group_sizes.max(), original_size // n_groups))
            # Add absolute maximum cap at original size
            target_size = min(target_size, original_size // max(1, n_groups // 2))
            
            # Skip resampling if all groups are already roughly balanced
            if group_sizes.max() - group_sizes.min() < target_size * 0.1:  # Allow 10% variance
                print("Dataset is already sufficiently balanced. Skipping resampling.")
                balanced_data = balanced_data.drop('temp_group', axis=1)
            else:
                # Resample groups to match target size
                resampled_groups = []
                for group_val in group_sizes.index:
                    group_data = balanced_data[balanced_data['temp_group'] == group_val]
                    
                    if len(group_data) < target_size:
                        resampled_group = resample(
                            group_data,
                            replace=True,
                            n_samples=target_size,
                            random_state=42
                        )
                        resampled_groups.append(resampled_group)
                    elif len(group_data) > target_size:
                        # Downsample large groups
                        resampled_group = resample(
                            group_data,
                            replace=False,
                            n_samples=target_size,
                            random_state=42
                        )
                        resampled_groups.append(resampled_group)
                    else:
                        resampled_groups.append(group_data)

                # Combine resampled groups and remove temporary column
                balanced_data = pd.concat(resampled_groups, ignore_index=True)
                balanced_data = balanced_data.drop('temp_group', axis=1)

        print("Resampling applied:")
        print(balanced_data[sensitive_attrs].nunique())  # Check unique values per sensitive attribute
        print("Final dataset shape:", balanced_data.shape)
        print("Group sizes before resampling:\n", group_sizes)
        print("Target group size:", target_size)
        print("New dataset shape after resampling:", balanced_data.shape)

        return balanced_data
    
    

    def generate_synthetic_data(self, random_state=50):
        """Generate synthetic data using ADASYN with a controlled sampling strategy."""

        sensitive_attrs = self._identify_sensitive_attributes()
        target = self._identify_target_variable()

        if target is None:
            return self.dataset.copy()

        # Prepare data
        X = self.dataset.drop(columns=[target])
        y = self.dataset[target]

        # Identify categorical columns
        categorical_columns = X.select_dtypes(include=['object', 'category']).columns.tolist()

        if categorical_columns:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            X_encoded = encoder.fit_transform(X[categorical_columns])
            X_encoded_df = pd.DataFrame(X_encoded, columns=encoder.get_feature_names_out(categorical_columns))

            # Drop original categorical columns and merge with encoded data
            X_numeric = X.drop(columns=categorical_columns).reset_index(drop=True)
            X_final = pd.concat([X_numeric, X_encoded_df], axis=1)
        else:
            X_final = X.reset_index(drop=True)

        # Normalize numerical features
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X_final)

        # Print class distribution before applying ADASYN
        print("Class distribution before ADASYN:")
        print(y.value_counts())

        # Dynamically adjust n_neighbors based on dataset size
        min_class_size = min(y.value_counts())
        n_neighbors = min(3, max(1, min_class_size - 1))  # Ensure n_neighbors is valid

        # Apply ADASYN only if class imbalance exists
        if len(y.value_counts()) > 1:
            adasyn = ADASYN(sampling_strategy="minority", random_state=random_state, n_neighbors=n_neighbors)
            X_resampled, y_resampled = adasyn.fit_resample(X_scaled, y)
        else:
            print("Dataset is already balanced. Returning original dataset.")
            return self.dataset

        # Convert synthetic data back to original format
        synthetic_data = pd.DataFrame(scaler.inverse_transform(X_resampled), columns=X_final.columns)
        synthetic_data[target] = y_resampled  # Assign generated labels

        # Print class distribution after ADASYN
        print("Class distribution after ADASYN:")
        print(pd.Series(y_resampled).value_counts())

        print("Synthetic data generation applied:")
        print(f"Original dataset shape: {self.dataset.shape}")
        print(f"Mitigated dataset shape: {synthetic_data.shape}")  # Confirm size change

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