"""
Dataset pipeline and data loaders
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class DataLoader:
    """
    Base class for data loaders.
    """
    
    def __init__(self, test_size: float = 0.2, val_size: float = 0.1, random_state: int = 42):
        """
        Initialize data loader.
        
        Args:
            test_size: Proportion of data for testing
            val_size: Proportion of training data for validation
            random_state: Random seed
        """
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.scaler = StandardScaler()
    
    def load(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load and split data.
        
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        raise NotImplementedError
    
    def _split_data(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into train/val/test.
        
        Args:
            X: Features
            y: Labels
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # First split: train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )
        
        # Second split: train/val
        val_size_adjusted = self.val_size / (1 - self.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train,
            test_size=val_size_adjusted,
            random_state=self.random_state,
            stratify=y_train,
        )
        
        # Scale features
        X_train = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns,
        )
        X_val = pd.DataFrame(
            self.scaler.transform(X_val),
            columns=X_val.columns,
        )
        X_test = pd.DataFrame(
            self.scaler.transform(X_test),
            columns=X_test.columns,
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test


class HeartDiseaseLoader(DataLoader):
    """
    Heart Disease UCI dataset loader.
    """
    
    def load(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load heart disease dataset.
        
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # Create synthetic heart disease data for demonstration
        np.random.seed(self.random_state)
        n_samples = 500
        
        # Generate synthetic features
        features = {
            'age': np.random.randint(30, 75, n_samples),
            'sex': np.random.binomial(1, 0.5, n_samples),
            'cp': np.random.randint(0, 4, n_samples),
            'trestbps': np.random.randint(90, 180, n_samples),
            'chol': np.random.randint(125, 400, n_samples),
            'fbs': np.random.binomial(1, 0.2, n_samples),
            'restecg': np.random.randint(0, 3, n_samples),
            'thalach': np.random.randint(60, 200, n_samples),
            'exang': np.random.binomial(1, 0.3, n_samples),
            'oldpeak': np.random.uniform(0, 6, n_samples),
            'slope': np.random.randint(0, 3, n_samples),
            'ca': np.random.randint(0, 4, n_samples),
            'thal': np.random.randint(0, 3, n_samples),
        }
        
        X = pd.DataFrame(features)
        
        # Create target based on simple rules
        y = (
            (X['age'] > 55).astype(int) +
            (X['chol'] > 250).astype(int) +
            (X['trestbps'] > 140).astype(int)
        ) >= 2
        y = y.astype(int).values
        
        return self._split_data(X, y)


class SyntheticBinaryClassificationLoader(DataLoader):
    """
    Synthetic binary classification dataset.
    """
    
    def load(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load synthetic dataset.
        
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # Use UCI breast cancer dataset
        data = load_breast_cancer()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = data.target
        
        # Downsample for faster processing
        X = X.sample(n=300, random_state=self.random_state)
        y = y[:len(X)]
        
        return self._split_data(X, y)


class DatasetFactory:
    """
    Factory for creating data loaders.
    """
    
    _loaders = {
        'heart_disease': HeartDiseaseLoader,
        'synthetic': SyntheticBinaryClassificationLoader,
    }
    
    @classmethod
    def create(
        cls,
        dataset_name: str,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
    ) -> DataLoader:
        """
        Create a data loader.
        
        Args:
            dataset_name: Name of dataset
            test_size: Proportion for testing
            val_size: Proportion for validation
            random_state: Random seed
            
        Returns:
            DataLoader instance
        """
        if dataset_name not in cls._loaders:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        loader_class = cls._loaders[dataset_name]
        return loader_class(test_size, val_size, random_state)
    
    @classmethod
    def register(cls, name: str, loader_class: type) -> None:
        """
        Register a new data loader.
        
        Args:
            name: Dataset name
            loader_class: DataLoader subclass
        """
        cls._loaders[name] = loader_class
