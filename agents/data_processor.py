import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import streamlit as st
import json
from pathlib import Path

class DataProcessor:
    def __init__(self):
        self.data = None
        self.metadata = {}
        self.data_profile = {}
    
    def load_file(self, uploaded_file) -> Tuple[bool, str]:
        """Load and process uploaded file"""
        try:
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            if file_extension == 'csv':
                self.data = pd.read_csv(uploaded_file)
            elif file_extension in ['xlsx', 'xls']:
                self.data = pd.read_excel(uploaded_file)
            elif file_extension == 'json':
                self.data = pd.read_json(uploaded_file)
            elif file_extension == 'parquet':
                self.data = pd.read_parquet(uploaded_file)
            else:
                return False, f"Unsupported file format: {file_extension}"
            
            # Fix pandas nullable dtypes for JSON serialization
            self._fix_dtypes()
            
            self._generate_metadata()
            self._profile_data()
            return True, "File loaded successfully!"
            
        except Exception as e:
            return False, f"Error loading file: {str(e)}"
    
    def _fix_dtypes(self):
        """Convert pandas nullable dtypes to standard dtypes for JSON serialization"""
        if self.data is not None:
            # Convert nullable integer types
            for col in self.data.columns:
                if str(self.data[col].dtype).startswith('Int'):
                    self.data[col] = self.data[col].astype('float64')
                elif str(self.data[col].dtype).startswith('Float'):
                    self.data[col] = self.data[col].astype('float64')
                elif str(self.data[col].dtype) == 'boolean':
                    self.data[col] = self.data[col].astype('bool')
                elif str(self.data[col].dtype) == 'string':
                    self.data[col] = self.data[col].astype('object')
    
    def _generate_metadata(self):
        """Generate metadata about the dataset"""
        if self.data is not None:
            # Convert dtypes to serializable format
            dtypes_serializable = {}
            for col, dtype in self.data.dtypes.items():
                dtypes_serializable[col] = str(dtype)
            
            # Convert null counts to int (not numpy int64)
            null_counts_serializable = {}
            for col, count in self.data.isnull().sum().items():
                null_counts_serializable[col] = int(count)
            
            self.metadata = {
                'shape': self.data.shape,
                'columns': list(self.data.columns),
                'dtypes': dtypes_serializable,
                'memory_usage': int(self.data.memory_usage(deep=True).sum()),
                'null_counts': null_counts_serializable,
                'duplicate_rows': int(self.data.duplicated().sum())
            }
    
    def _profile_data(self):
        """Generate comprehensive data profile"""
        if self.data is not None:
            profile = {}
            
            for col in self.data.columns:
                col_profile = {
                    'dtype': str(self.data[col].dtype),
                    'null_count': int(self.data[col].isnull().sum()),
                    'null_percentage': float((self.data[col].isnull().sum() / len(self.data)) * 100),
                    'unique_count': int(self.data[col].nunique()),
                    'unique_percentage': float((self.data[col].nunique() / len(self.data)) * 100)
                }
                
                if pd.api.types.is_numeric_dtype(self.data[col]):
                    # Convert all numeric values to standard Python types
                    col_profile.update({
                        'mean': float(self.data[col].mean()) if not pd.isna(self.data[col].mean()) else None,
                        'median': float(self.data[col].median()) if not pd.isna(self.data[col].median()) else None,
                        'std': float(self.data[col].std()) if not pd.isna(self.data[col].std()) else None,
                        'min': float(self.data[col].min()) if not pd.isna(self.data[col].min()) else None,
                        'max': float(self.data[col].max()) if not pd.isna(self.data[col].max()) else None,
                        'q25': float(self.data[col].quantile(0.25)) if not pd.isna(self.data[col].quantile(0.25)) else None,
                        'q75': float(self.data[col].quantile(0.75)) if not pd.isna(self.data[col].quantile(0.75)) else None,
                        'skewness': float(self.data[col].skew()) if not pd.isna(self.data[col].skew()) else None,
                        'kurtosis': float(self.data[col].kurtosis()) if not pd.isna(self.data[col].kurtosis()) else None
                    })
                elif pd.api.types.is_datetime64_any_dtype(self.data[col]):
                    col_profile.update({
                        'min_date': str(self.data[col].min()) if not pd.isna(self.data[col].min()) else None,
                        'max_date': str(self.data[col].max()) if not pd.isna(self.data[col].max()) else None,
                        'date_range': str(self.data[col].max() - self.data[col].min()) if not pd.isna(self.data[col].min()) else None
                    })
                else:  # Categorical/Object
                    mode_val = self.data[col].mode()
                    value_counts = self.data[col].value_counts()
                    
                    col_profile.update({
                        'most_frequent': str(mode_val.iloc[0]) if not mode_val.empty else None,
                        'most_frequent_count': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0
                    })
                
                profile[col] = col_profile
            
            self.data_profile = profile
    
    def get_sample_data(self, n: int = 100) -> pd.DataFrame:
        """Get sample of data for display"""
        if self.data is not None:
            return self.data.head(n)
        return pd.DataFrame()
    
    def get_column_info(self) -> Dict[str, Any]:
        """Get detailed column information"""
        return self.data_profile
    
    def get_numeric_columns(self) -> List[str]:
        """Get list of numeric columns"""
        if self.data is not None:
            return list(self.data.select_dtypes(include=[np.number]).columns)
        return []
    
    def get_categorical_columns(self) -> List[str]:
        """Get list of categorical columns"""
        if self.data is not None:
            return list(self.data.select_dtypes(include=['object', 'category']).columns)
        return []
    
    def get_datetime_columns(self) -> List[str]:
        """Get list of datetime columns"""
        if self.data is not None:
            return list(self.data.select_dtypes(include=['datetime64']).columns)
        return []
    
    def prepare_for_visualization(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare DataFrame for Plotly visualization by converting dtypes"""
        df_viz = df.copy()
        
        for col in df_viz.columns:
            if pd.api.types.is_numeric_dtype(df_viz[col]):
                # Convert to standard float64
                df_viz[col] = pd.to_numeric(df_viz[col], errors='coerce').astype('float64')
            elif pd.api.types.is_datetime64_any_dtype(df_viz[col]):
                # Ensure datetime format
                df_viz[col] = pd.to_datetime(df_viz[col], errors='coerce')
            else:
                # Convert to string
                df_viz[col] = df_viz[col].astype('str')
        
        # Remove any remaining NaN values that might cause issues
        df_viz = df_viz.fillna('N/A')
        
        return df_viz
