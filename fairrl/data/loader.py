import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class AdultDataset:
    """
    Class to load and preprocess the UCI Adult dataset for RL tasks.
    """
    def __init__(self, data_path=None):
        self.data_path = data_path or "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
        self.column_names = [
            "age", "workclass", "fnlwgt", "education", "education-num",
            "marital-status", "occupation", "relationship", "race",
            "sex", "capital-gain", "capital-loss", "hours-per-week",
            "native-country", "income"
        ]
        self.categorical_features = [
            "workclass", "education", "marital-status", "occupation",
            "relationship", "race", "native-country"
        ]
        self.numerical_features = [
            "age", "fnlwgt", "education-num", "capital-gain",
            "capital-loss", "hours-per-week"
        ]
        self.sensitive_attribute = "sex"
        self.target = "income"

    def load_data(self):
        """Load the raw dataset"""
        df = pd.read_csv(self.data_path, header=None, names=self.column_names,
                          na_values="?", skipinitialspace=True)
        # Clean data
        df = df.dropna()
        # Binary encoding for target variable
        df[self.target] = df[self.target].apply(lambda x: 1 if x.strip() == ">50K" else 0)
        # Binary encoding for sensitive attribute
        df[self.sensitive_attribute] = df[self.sensitive_attribute].apply(lambda x: 1 if x.strip() == "Male" else 0)

        return df

    def preprocess_data(self, df=None):
        """Preprocess the data for ML/RL models"""
        if df is None:
            df = self.load_data()

        # Create preprocessing pipeline
        numerical_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, self.numerical_features),
                ('cat', categorical_transformer, self.categorical_features)
            ])

        # Split features and target
        X = df.drop(columns=[self.target])
        y = df[self.target]

        # Extract sensitive attribute before preprocessing
        sensitive = X[self.sensitive_attribute].copy()

        # Remove sensitive attribute from features to be preprocessed
        # We'll keep it separately as a binary feature
        X_for_preprocessing = X.drop(columns=[self.sensitive_attribute])

        # Apply preprocessing to features excluding sensitive attribute
        X_processed = preprocessor.fit_transform(X_for_preprocessing)

        # Convert to dense array if it's sparse
        if hasattr(X_processed, 'toarray'):
            X_processed = X_processed.toarray()

        # Add sensitive attribute as the last feature
        sensitive_array = sensitive.values.reshape(-1, 1)
        X_processed_with_sensitive = np.hstack((X_processed, sensitive_array))

        # Return preprocessed data and original sensitive attribute
        return X_processed_with_sensitive, y, sensitive, preprocessor

    def get_feature_names(self, preprocessor):
        """Get feature names after preprocessing"""
        numeric_features = self.numerical_features

        # Get one-hot encoded feature names
        categorical_features = []
        for feature, categories in zip(
            self.categorical_features,
            preprocessor.named_transformers_['cat'].named_steps['onehot'].categories_
        ):
            for category in categories:
                categorical_features.append(f"{feature}_{category}")

        # Add sensitive attribute as the last feature
        all_features = numeric_features + categorical_features + [self.sensitive_attribute]
        return all_features

if __name__ == "__main__":
    # Example usage
    dataset = AdultDataset()
    df = dataset.load_data()
    print("Dataset shape:", df.shape)
    print("Sample data:")
    print(df.head())

    # Check distribution of sensitive attribute and target
    print("\nSensitive attribute distribution:")
    print(df["sex"].value_counts(normalize=True))

    print("\nTarget distribution:")
    print(df["income"].value_counts(normalize=True))

    # Check potential bias
    print("\nIncome distribution by gender:")
    print(pd.crosstab(df["sex"], df["income"], normalize="index"))