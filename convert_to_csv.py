import pandas as pd

# Define the column names based on the dataset documentation
column_names = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
]

# Read the data file
data = pd.read_csv("adult.data", header=None, names=column_names, na_values=" ?", skipinitialspace=True)

# Save the data to a CSV file
data.to_csv("adult.csv", index=False)

print("Data has been converted to adult.csv")