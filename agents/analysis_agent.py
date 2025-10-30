import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

class AnalysisAgent:
    def __init__(self, data: pd.DataFrame):
        self.data = data
    
    def descriptive_analysis(self, columns: list = None) -> dict:
        """Perform descriptive analysis on specified columns"""
        if columns is None:
            columns = self.data.select_dtypes(include=[np.number]).columns.tolist()
        
        results = {}
        for col in columns:
            if col in self.data.columns:
                if pd.api.types.is_numeric_dtype(self.data[col]):
                    results[col] = {
                        'count': len(self.data[col].dropna()),
                        'mean': self.data[col].mean(),
                        'median': self.data[col].median(),
                        'mode': self.data[col].mode().iloc[0] if not self.data[col].mode().empty else None,
                        'std': self.data[col].std(),
                        'min': self.data[col].min(),
                        'max': self.data[col].max(),
                        'q25': self.data[col].quantile(0.25),
                        'q75': self.data[col].quantile(0.75),
                        'skewness': self.data[col].skew(),
                        'kurtosis': self.data[col].kurtosis(),
                        'outliers_count': len(self._detect_outliers(self.data[col]))
                    }
                else:
                    results[col] = {
                        'count': len(self.data[col].dropna()),
                        'unique': self.data[col].nunique(),
                        'top_value': self.data[col].mode().iloc[0] if not self.data[col].mode().empty else None,
                        'top_freq': self.data[col].value_counts().iloc[0] if len(self.data[col].value_counts()) > 0 else 0
                    }
        
        return results
    
    def groupby_analysis(self, target_col: str, groupby_col: str, agg_func: str = 'mean') -> pd.DataFrame:
        """Perform groupby analysis"""
        try:
            if agg_func == 'mean':
                result = self.data.groupby(groupby_col)[target_col].mean().reset_index()
            elif agg_func == 'sum':
                result = self.data.groupby(groupby_col)[target_col].sum().reset_index()
            elif agg_func == 'count':
                result = self.data.groupby(groupby_col)[target_col].count().reset_index()
            elif agg_func == 'min':
                result = self.data.groupby(groupby_col)[target_col].min().reset_index()
            elif agg_func == 'max':
                result = self.data.groupby(groupby_col)[target_col].max().reset_index()
            else:
                result = self.data.groupby(groupby_col)[target_col].mean().reset_index()
            
            return result.sort_values(target_col, ascending=False)
        except Exception as e:
            raise Exception(f"Error in groupby analysis: {str(e)}")
    
    def correlation_analysis(self, columns: list = None) -> pd.DataFrame:
        """Perform correlation analysis"""
        if columns is None:
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        else:
            numeric_cols = [col for col in columns if col in self.data.columns and pd.api.types.is_numeric_dtype(self.data[col])]
        
        if len(numeric_cols) < 2:
            raise ValueError("Need at least 2 numeric columns for correlation analysis")
        
        return self.data[numeric_cols].corr()
    
    def trend_analysis(self, value_col: str, time_col: str) -> dict:
        """Perform trend analysis over time"""
        try:
            # Convert time column to datetime if not already
            if not pd.api.types.is_datetime64_any_dtype(self.data[time_col]):
                temp_data = self.data.copy()
                temp_data[time_col] = pd.to_datetime(temp_data[time_col])
            else:
                temp_data = self.data.copy()
            
            # Sort by time
            temp_data = temp_data.sort_values(time_col)
            
            # Calculate trend using linear regression
            X = np.arange(len(temp_data)).reshape(-1, 1)
            y = temp_data[value_col].values
            
            model = LinearRegression()
            model.fit(X, y)
            
            trend_slope = model.coef_[0]
            r2 = r2_score(y, model.predict(X))
            
            # Monthly/daily aggregation
            temp_data['period'] = temp_data[time_col].dt.to_period('M')
            trend_data = temp_data.groupby('period')[value_col].mean().reset_index()
            trend_data['period'] = trend_data['period'].astype(str)
            
            return {
                'trend_slope': trend_slope,
                'r_squared': r2,
                'trend_direction': 'increasing' if trend_slope > 0 else 'decreasing',
                'trend_strength': 'strong' if abs(r2) > 0.7 else 'moderate' if abs(r2) > 0.3 else 'weak',
                'trend_data': trend_data
            }
        except Exception as e:
            raise Exception(f"Error in trend analysis: {str(e)}")
    
    def statistical_test(self, col1: str, col2: str = None, test_type: str = 'ttest') -> dict:
        """Perform statistical tests"""
        try:
            if test_type == 'ttest' and col2:
                # Independent t-test
                stat, p_value = stats.ttest_ind(self.data[col1].dropna(), self.data[col2].dropna())
                test_name = "Independent T-Test"
            elif test_type == 'normality':
                # Shapiro-Wilk normality test
                stat, p_value = stats.shapiro(self.data[col1].dropna().sample(min(5000, len(self.data[col1].dropna()))))
                test_name = "Shapiro-Wilk Normality Test"
            else:
                return {"error": "Unsupported test type or missing parameters"}
            
            return {
                'test_name': test_name,
                'statistic': stat,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'interpretation': f"{'Reject' if p_value < 0.05 else 'Fail to reject'} null hypothesis (α = 0.05)"
            }
        except Exception as e:
            return {"error": f"Statistical test failed: {str(e)}"}
    
    def clustering_analysis(self, columns: list, n_clusters: int = 3) -> dict:
        """Perform K-means clustering"""
        try:
            # Select numeric columns only
            numeric_cols = [col for col in columns if col in self.data.columns and pd.api.types.is_numeric_dtype(self.data[col])]
            
            if len(numeric_cols) < 2:
                raise ValueError("Need at least 2 numeric columns for clustering")
            
            # Prepare data
            cluster_data = self.data[numeric_cols].dropna()
            
            # Standardize features
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(cluster_data)
            
            # Perform clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(scaled_data)
            
            # Add clusters to original data
            result_data = cluster_data.copy()
            result_data['Cluster'] = clusters
            
            # Calculate cluster centers in original scale
            cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
            centers_df = pd.DataFrame(cluster_centers, columns=numeric_cols)
            centers_df['Cluster'] = range(n_clusters)
            
            return {
                'data_with_clusters': result_data,
                'cluster_centers': centers_df,
                'inertia': kmeans.inertia_,
                'n_clusters': n_clusters
            }
        except Exception as e:
            raise Exception(f"Clustering analysis failed: {str(e)}")
    
    def _detect_outliers(self, series: pd.Series) -> list:
        """Detect outliers using IQR method"""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = series[(series < lower_bound) | (series > upper_bound)]
        return outliers.index.tolist()
