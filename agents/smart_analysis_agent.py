"""
Smart Analysis Agent - Automatically suggests best analyses for the dataset
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
from groq import Groq
import os


class SmartAnalysisAgent:
    """Agent that intelligently analyzes data and suggests best insights"""
    
    def __init__(self, data: pd.DataFrame, groq_api_key: str = None):
        self.data = data
        self.groq_api_key = groq_api_key or os.getenv('GROQ_API_KEY')
        self.client = Groq(api_key=self.groq_api_key) if self.groq_api_key else None
        
        self.numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
        self.datetime_cols = data.select_dtypes(include=['datetime64']).columns.tolist()
    
    def analyze_data_structure(self) -> Dict[str, Any]:
        """Analyze the structure and characteristics of the dataset"""
        
        structure = {
            'total_rows': len(self.data),
            'total_columns': len(self.data.columns),
            'numeric_columns': self.numeric_cols,
            'categorical_columns': self.categorical_cols,
            'datetime_columns': self.datetime_cols,
            'missing_data': {},
            'unique_ratios': {},
            'high_cardinality_cols': [],
            'potential_id_cols': [],
            'potential_target_cols': []
        }
        
        # Analyze each column
        for col in self.data.columns:
            null_pct = (self.data[col].isnull().sum() / len(self.data)) * 100
            unique_ratio = self.data[col].nunique() / len(self.data)
            
            structure['missing_data'][col] = round(null_pct, 2)
            structure['unique_ratios'][col] = round(unique_ratio, 4)
            
            # Identify high cardinality categorical columns
            if col in self.categorical_cols and self.data[col].nunique() > 20:
                structure['high_cardinality_cols'].append(col)
            
            # Identify potential ID columns
            if unique_ratio > 0.95:
                structure['potential_id_cols'].append(col)
            
            # Identify potential target columns (numeric with reasonable unique values)
            if col in self.numeric_cols and 2 <= self.data[col].nunique() <= 50:
                structure['potential_target_cols'].append(col)
        
        return structure
    
    def generate_smart_suggestions(self) -> List[Dict[str, Any]]:
        """Generate top 3 analysis suggestions based on data characteristics"""
        
        structure = self.analyze_data_structure()
        suggestions = []
        
        # Suggestion 1: Distribution analysis of key numeric columns
        if len(self.numeric_cols) > 0:
            # Find most interesting numeric columns (high variance, no ID-like patterns)
            interesting_numeric = [
                col for col in self.numeric_cols 
                if col not in structure['potential_id_cols'] 
                and structure['unique_ratios'].get(col, 0) > 0.01
            ]
            
            if interesting_numeric:
                top_numeric = interesting_numeric[:3]
                
                suggestions.append({
                    'title': '📊 Distribution Analysis of Key Variables',
                    'description': f'Understanding the spread and patterns in your main numeric variables: {", ".join(top_numeric)}',
                    'type': 'distribution',
                    'columns': top_numeric,
                    'priority': 1,
                    'explanation': f'These columns show significant variation in your data. Analyzing their distributions helps identify:\n'
                                 f'• Typical value ranges and outliers\n'
                                 f'• Whether data is normally distributed or skewed\n'
                                 f'• Potential data quality issues',
                    'viz_type': 'dashboard_histogram'
                })
        
        # Suggestion 2: Relationship/Correlation analysis
        if len(self.numeric_cols) >= 2:
            # Find numeric columns with interesting relationships
            corr_matrix = self.data[self.numeric_cols].corr()
            
            # Find strongest correlations
            strong_corrs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = abs(corr_matrix.iloc[i, j])
                    if corr_val > 0.3 and corr_val < 0.99:  # Avoid perfect correlations (likely duplicates)
                        strong_corrs.append((
                            corr_matrix.columns[i],
                            corr_matrix.columns[j],
                            corr_matrix.iloc[i, j]
                        ))
            
            if strong_corrs:
                strong_corrs.sort(key=lambda x: abs(x[2]), reverse=True)
                top_pair = strong_corrs[0]
                
                suggestions.append({
                    'title': '🔗 Relationship Analysis Between Variables',
                    'description': f'Exploring how {top_pair[0]} relates to {top_pair[1]} and other variables',
                    'type': 'correlation',
                    'columns': [top_pair[0], top_pair[1]],
                    'all_numeric': self.numeric_cols,
                    'priority': 2,
                    'explanation': f'Found a {"strong" if abs(top_pair[2]) > 0.7 else "moderate"} '
                                 f'{"positive" if top_pair[2] > 0 else "negative"} correlation ({top_pair[2]:.3f}) between '
                                 f'{top_pair[0]} and {top_pair[1]}. Understanding relationships helps:\n'
                                 f'• Identify predictive patterns\n'
                                 f'• Find redundant variables\n'
                                 f'• Discover hidden insights',
                    'viz_type': 'scatter_and_heatmap'
                })
            else:
                # If no strong correlations, still show correlation matrix
                suggestions.append({
                    'title': '🔗 Variable Correlation Overview',
                    'description': 'Comprehensive view of all variable relationships',
                    'type': 'correlation',
                    'columns': self.numeric_cols[:5],
                    'all_numeric': self.numeric_cols,
                    'priority': 2,
                    'explanation': 'No strong correlations detected, but analyzing the full correlation matrix helps:\n'
                                 '• Confirm variable independence\n'
                                 '• Identify weak patterns\n'
                                 '• Guide feature selection',
                    'viz_type': 'correlation_matrix'
                })
        
        # Suggestion 3: Category comparison (if categorical columns exist)
        if len(self.categorical_cols) > 0 and len(self.numeric_cols) > 0:
            # Find best categorical column (reasonable cardinality)
            good_categorical = [
                col for col in self.categorical_cols
                if 2 <= self.data[col].nunique() <= 15
            ]
            
            if good_categorical:
                cat_col = good_categorical[0]
                # Find best numeric column to analyze by this category
                num_col = self.numeric_cols[0]
                
                suggestions.append({
                    'title': f'📈 Comparing {num_col} Across {cat_col} Categories',
                    'description': f'How {num_col} varies across different {cat_col} groups',
                    'type': 'groupby',
                    'columns': [num_col, cat_col],
                    'priority': 3,
                    'explanation': f'Analyzing {num_col} by {cat_col} reveals:\n'
                                 f'• Which categories have highest/lowest values\n'
                                 f'• Performance differences between groups\n'
                                 f'• Patterns that might be hidden in aggregate data',
                    'viz_type': 'bar_and_box'
                })
        
        # Suggestion 4: Time series analysis (if datetime columns exist)
        if len(self.datetime_cols) > 0 and len(self.numeric_cols) > 0:
            date_col = self.datetime_cols[0]
            num_col = self.numeric_cols[0]
            
            suggestions.append({
                'title': f'📅 Trend Analysis: {num_col} Over Time',
                'description': f'Tracking how {num_col} changes over time',
                'type': 'timeseries',
                'columns': [num_col, date_col],
                'priority': 2 if len(suggestions) < 3 else 4,
                'explanation': f'Time-based analysis of {num_col} shows:\n'
                             f'• Long-term trends and patterns\n'
                             f'• Seasonal variations\n'
                             f'• Recent changes and anomalies',
                'viz_type': 'line_chart'
            })
        
        # Suggestion 5: Outlier detection
        if len(self.numeric_cols) > 0:
            # Find columns with potential outliers
            outlier_cols = []
            for col in self.numeric_cols[:3]:
                Q1 = self.data[col].quantile(0.25)
                Q3 = self.data[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = self.data[
                    (self.data[col] < (Q1 - 1.5 * IQR)) | 
                    (self.data[col] > (Q3 + 1.5 * IQR))
                ]
                if len(outliers) > 0:
                    outlier_cols.append((col, len(outliers)))
            
            if outlier_cols and len(suggestions) < 3:
                outlier_cols.sort(key=lambda x: x[1], reverse=True)
                col_name, outlier_count = outlier_cols[0]
                
                suggestions.append({
                    'title': f'🎯 Outlier Analysis in {col_name}',
                    'description': f'Investigating {outlier_count} unusual values in {col_name}',
                    'type': 'outliers',
                    'columns': [col_name],
                    'priority': 3,
                    'explanation': f'Detected {outlier_count} outliers in {col_name}. Outlier analysis helps:\n'
                                 f'• Identify data quality issues\n'
                                 f'• Discover exceptional cases\n'
                                 f'• Understand data range and variability',
                    'viz_type': 'box_plot'
                })
        
        # Sort by priority and return top 3
        suggestions.sort(key=lambda x: x['priority'])
        return suggestions[:3]
    
    def generate_ai_explanation(self, suggestion: Dict[str, Any]) -> str:
        """Generate AI-powered explanation for a suggestion"""
        
        if not self.client:
            return suggestion.get('explanation', 'No explanation available')
        
        # Prepare data summary for LLM
        data_summary = f"""
Dataset Overview:
- Total rows: {len(self.data)}
- Columns: {', '.join(self.data.columns.tolist())}
- Numeric columns: {', '.join(self.numeric_cols)}
- Categorical columns: {', '.join(self.categorical_cols)}

Suggestion Type: {suggestion['type']}
Columns Involved: {', '.join(suggestion['columns'])}
        """
        
        prompt = f"""You are a data analyst explaining insights to a business user.

{data_summary}

Provide a brief, clear explanation (3-4 sentences) about why this analysis is valuable and what insights they might gain. 
Focus on business value, not technical details.

Analysis: {suggestion['title']}
"""
        
        try:
            response = self.client.chat.completions.create(
                model="mixtral-8x7b-32768",
                messages=[
                    {"role": "system", "content": "You are a helpful data analyst who explains insights clearly to non-technical users."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=300
            )
            
            return response.choices[0].message.content
        except Exception as e:
            return suggestion.get('explanation', 'No explanation available')
