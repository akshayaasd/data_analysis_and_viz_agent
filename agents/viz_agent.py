import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List


class VizAgent:
    def __init__(self):
        self.default_height = 600
        self.default_width = 1000
        self.color_palette = px.colors.qualitative.Set3
    
    def _prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for Plotly visualization"""
        df_viz = data.copy()
        
        # Convert pandas nullable dtypes to standard types
        for col in df_viz.columns:
            dtype_str = str(df_viz[col].dtype)
            
            if dtype_str.startswith('Int') or dtype_str.startswith('Float'):
                df_viz[col] = pd.to_numeric(df_viz[col], errors='coerce').astype('float64')
            elif dtype_str == 'boolean':
                df_viz[col] = df_viz[col].astype('bool')
            elif dtype_str == 'string':
                df_viz[col] = df_viz[col].astype('object')
            elif pd.api.types.is_datetime64_any_dtype(df_viz[col]):
                df_viz[col] = pd.to_datetime(df_viz[col], errors='coerce')
        
        # Handle NaN values
        for col in df_viz.columns:
            if df_viz[col].dtype == 'float64':
                df_viz[col] = df_viz[col].fillna(0)
            elif df_viz[col].dtype == 'object':
                df_viz[col] = df_viz[col].fillna('Unknown')
            elif df_viz[col].dtype == 'bool':
                df_viz[col] = df_viz[col].fillna(False)
        
        return df_viz
    
    def create_visualization(self, data: pd.DataFrame, viz_type: str, **kwargs) -> go.Figure:
        """Create visualization based on type and parameters"""
        viz_data = self._prepare_data(data)
        
        # Limit data for better visualization performance
        if len(viz_data) > 1000 and viz_type in ['scatter', 'line']:
            viz_data = viz_data.sample(n=1000, random_state=42)
        elif len(viz_data) > 50 and viz_type in ['bar']:
            if 'y' in kwargs:
                viz_data = viz_data.nlargest(20, kwargs['y'])
        
        viz_methods = {
            'bar': self._create_bar_chart,
            'line': self._create_line_chart,
            'scatter': self._create_scatter_plot,
            'histogram': self._create_histogram,
            'box': self._create_box_plot,
            'violin': self._create_violin_plot,
            'pie': self._create_pie_chart,
            'heatmap': self._create_heatmap,
            'area': self._create_area_chart,
            'correlation': self._create_correlation_matrix
        }
        
        if viz_type not in viz_methods:
            raise ValueError(f"Unsupported visualization type: {viz_type}")
        
        return viz_methods[viz_type](viz_data, **kwargs)
    
    def _create_bar_chart(self, data: pd.DataFrame, x: str, y: str, color: str = None, **kwargs) -> go.Figure:
        """Create enhanced bar chart"""
        fig = px.bar(
            data, 
            x=x, 
            y=y, 
            color=color,
            title=kwargs.get('title', f'{y.replace("_", " ").title()} by {x.replace("_", " ").title()}'),
            height=kwargs.get('height', self.default_height),
            color_discrete_sequence=self.color_palette
        )
        
        fig.update_layout(
            xaxis_title=x.replace('_', ' ').title(),
            yaxis_title=y.replace('_', ' ').title(),
            showlegend=color is not None,
            font=dict(size=12),
            title_font_size=16,
            margin=dict(l=80, r=80, t=100, b=80),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        if len(data) > 10:
            fig.update_xaxes(tickangle=45)
        
        fig.update_traces(texttemplate='%{y:.2s}', textposition='outside')
        
        return fig
    
    def _create_line_chart(self, data: pd.DataFrame, x: str, y: str, color: str = None, **kwargs) -> go.Figure:
        """Create enhanced line chart"""
        fig = px.line(
            data,
            x=x,
            y=y,
            color=color,
            title=kwargs.get('title', f'{y.replace("_", " ").title()} Trend over {x.replace("_", " ").title()}'),
            height=kwargs.get('height', self.default_height),
            markers=True,
            color_discrete_sequence=self.color_palette
        )
        
        fig.update_layout(
            xaxis_title=x.replace('_', ' ').title(),
            yaxis_title=y.replace('_', ' ').title(),
            font=dict(size=12),
            title_font_size=16,
            margin=dict(l=80, r=80, t=100, b=80),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            hovermode='x unified'
        )
        
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        
        return fig
    
    def _create_scatter_plot(self, data: pd.DataFrame, x: str, y: str, color: str = None, size: str = None, **kwargs) -> go.Figure:
        """Create enhanced scatter plot"""
        fig = px.scatter(
            data,
            x=x,
            y=y,
            color=color,
            size=size,
            title=kwargs.get('title', f'{y.replace("_", " ").title()} vs {x.replace("_", " ").title()}'),
            height=kwargs.get('height', self.default_height),
            trendline="ols" if kwargs.get('trendline', False) else None,
            color_discrete_sequence=self.color_palette
        )
        
        fig.update_layout(
            xaxis_title=x.replace('_', ' ').title(),
            yaxis_title=y.replace('_', ' ').title(),
            font=dict(size=12),
            title_font_size=16,
            margin=dict(l=80, r=80, t=100, b=80),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        fig.update_traces(marker=dict(opacity=0.7, line=dict(width=0.5, color='DarkSlateGrey')))
        
        return fig
    
    def _create_histogram(self, data: pd.DataFrame, x: str, color: str = None, **kwargs) -> go.Figure:
        """Create enhanced histogram"""
        fig = px.histogram(
            data,
            x=x,
            color=color,
            title=kwargs.get('title', f'Distribution of {x.replace("_", " ").title()}'),
            height=kwargs.get('height', self.default_height),
            nbins=kwargs.get('bins', min(30, len(data.dropna())//10 + 1)),
            color_discrete_sequence=self.color_palette
        )
        
        fig.update_layout(
            xaxis_title=x.replace('_', ' ').title(),
            yaxis_title='Count',
            font=dict(size=12),
            title_font_size=16,
            margin=dict(l=80, r=80, t=100, b=80),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            bargap=0.1
        )
        
        return fig
    
    def _create_box_plot(self, data: pd.DataFrame, x: str = None, y: str = None, color: str = None, **kwargs) -> go.Figure:
        """Create enhanced box plot"""
        fig = px.box(
            data,
            x=x,
            y=y,
            color=color,
            title=kwargs.get('title', f'Box Plot of {y.replace("_", " ").title() if y else "Values"}'),
            height=kwargs.get('height', self.default_height),
            points='outliers',
            color_discrete_sequence=self.color_palette
        )
        
        fig.update_layout(
            xaxis_title=x.replace('_', ' ').title() if x else '',
            yaxis_title=y.replace('_', ' ').title() if y else 'Value',
            font=dict(size=12),
            title_font_size=16,
            margin=dict(l=80, r=80, t=100, b=80),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        
        return fig
    
    def _create_violin_plot(self, data: pd.DataFrame, x: str = None, y: str = None, color: str = None, **kwargs) -> go.Figure:
        """Create enhanced violin plot"""
        fig = px.violin(
            data,
            x=x,
            y=y,
            color=color,
            box=True,
            points='outliers',
            title=kwargs.get('title', f'Violin Plot of {y.replace("_", " ").title() if y else "Values"}'),
            height=kwargs.get('height', self.default_height),
            color_discrete_sequence=self.color_palette
        )
        
        fig.update_layout(
            xaxis_title=x.replace('_', ' ').title() if x else '',
            yaxis_title=y.replace('_', ' ').title() if y else 'Value',
            font=dict(size=12),
            title_font_size=16,
            margin=dict(l=80, r=80, t=100, b=80),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            violinmode='overlay' if color else 'group'
        )
        
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        
        return fig
    
    def _create_pie_chart(self, data: pd.DataFrame, values: str, names: str, **kwargs) -> go.Figure:
        """Create enhanced pie chart"""
        if len(data) > 10:
            data = data.nlargest(10, values)
        
        pie_data = data.copy()
        pie_data[values] = pd.to_numeric(pie_data[values], errors='coerce').fillna(0)
        
        fig = px.pie(
            pie_data,
            values=values,
            names=names,
            title=kwargs.get('title', f'{values.replace("_", " ").title()} by {names.replace("_", " ").title()}'),
            height=kwargs.get('height', self.default_height),
            color_discrete_sequence=self.color_palette
        )
        
        fig.update_traces(
            textposition='inside', 
            textinfo='percent+label',
            hovertemplate='<b>%{label}</b><br>Value: %{value}<br>Percentage: %{percent}<extra></extra>'
        )
        
        fig.update_layout(
            font=dict(size=12),
            title_font_size=16,
            margin=dict(l=80, r=80, t=100, b=80)
        )
        
        return fig
    
    def _create_heatmap(self, data: pd.DataFrame, x: str = None, y: str = None, z: str = None, **kwargs) -> go.Figure:
        """Create enhanced heatmap"""
        if x and y and z:
            pivot_data = data.pivot_table(values=z, index=y, columns=x, aggfunc='mean')
            
            fig = go.Figure(data=go.Heatmap(
                z=pivot_data.values.astype('float64'),
                x=[col.replace('_', ' ').title() for col in pivot_data.columns],
                y=[idx.replace('_', ' ').title() if isinstance(idx, str) else idx for idx in pivot_data.index],
                colorscale='Viridis',
                text=np.round(pivot_data.values, 2),
                texttemplate='%{text}',
                textfont={"size": 10},
                colorbar=dict(title=z.replace('_', ' ').title()),
                hovertemplate='<b>%{x}</b><br>%{y}<br>Value: %{z:.2f}<extra></extra>'
            ))
            
            title = kwargs.get('title', f'{z.replace("_", " ").title()} Heatmap')
        else:
            numeric_data = data.select_dtypes(include=[np.number])
            
            for col in numeric_data.columns:
                numeric_data[col] = pd.to_numeric(numeric_data[col], errors='coerce')
            
            corr_matrix = numeric_data.corr()
            
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values.astype('float64'),
                x=[col.replace('_', ' ').title() for col in corr_matrix.columns],
                y=[col.replace('_', ' ').title() for col in corr_matrix.columns],
                colorscale='RdBu_r',
                zmid=0,
                text=np.round(corr_matrix.values, 2),
                texttemplate='%{text}',
                textfont={"size": 10},
                colorbar=dict(title="Correlation"),
                hovertemplate='<b>%{x}</b><br>%{y}<br>Correlation: %{z:.2f}<extra></extra>'
            ))
            
            title = kwargs.get('title', 'Correlation Heatmap')
        
        fig.update_layout(
            title=title,
            height=kwargs.get('height', self.default_height),
            font=dict(size=12),
            title_font_size=16,
            margin=dict(l=100, r=100, t=100, b=100)
        )
        
        return fig
    
    def _create_area_chart(self, data: pd.DataFrame, x: str, y: str, color: str = None, **kwargs) -> go.Figure:
        """Create enhanced area chart"""
        fig = px.area(
            data,
            x=x,
            y=y,
            color=color,
            title=kwargs.get('title', f'{y.replace("_", " ").title()} Area Chart over {x.replace("_", " ").title()}'),
            height=kwargs.get('height', self.default_height),
            color_discrete_sequence=self.color_palette
        )
        
        fig.update_layout(
            xaxis_title=x.replace('_', ' ').title(),
            yaxis_title=y.replace('_', ' ').title(),
            font=dict(size=12),
            title_font_size=16,
            margin=dict(l=80, r=80, t=100, b=80),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            hovermode='x unified'
        )
        
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_traces(opacity=0.7)
        
        return fig
    
    def _create_correlation_matrix(self, data: pd.DataFrame, **kwargs) -> go.Figure:
        """Create enhanced correlation matrix"""
        numeric_data = data.select_dtypes(include=[np.number])
        
        for col in numeric_data.columns:
            numeric_data[col] = pd.to_numeric(numeric_data[col], errors='coerce')
        
        corr_matrix = numeric_data.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values.astype('float64'),
            x=[col.replace('_', ' ').title() for col in corr_matrix.columns],
            y=[col.replace('_', ' ').title() for col in corr_matrix.columns],
            colorscale='RdBu_r',
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(title="Correlation Coefficient"),
            hoverongaps=False
        ))
        
        fig.update_layout(
            title=kwargs.get('title', 'Correlation Matrix'),
            height=kwargs.get('height', max(500, len(corr_matrix.columns) * 40)),
            width=kwargs.get('width', max(500, len(corr_matrix.columns) * 40)),
            font=dict(size=12),
            title_font_size=16,
            margin=dict(l=100, r=100, t=100, b=100)
        )
        
        return fig
    
    def create_dashboard(self, figures: List[go.Figure], titles: List[str] = None, rows: int = None, cols: int = None, **kwargs) -> go.Figure:
        """
        Create a dashboard by combining multiple figures into subplots
        
        Args:
            figures: List of Plotly figure objects to combine
            titles: List of titles for each subplot
            rows: Number of rows (auto-calculated if not provided)
            cols: Number of columns (default: 2)
            **kwargs: Additional arguments for make_subplots
        
        Returns:
            Combined figure with subplots
        """
        if not figures:
            raise ValueError("No figures provided to create dashboard")
        
        n_plots = len(figures)
        
        # Auto-calculate grid dimensions if not provided
        if cols is None:
            cols = 2
        if rows is None:
            rows = (n_plots + cols - 1) // cols  # Ceiling division
        
        # Create subplot titles
        if titles is None:
            titles = [f"Plot {i+1}" for i in range(n_plots)]
        elif len(titles) < n_plots:
            titles.extend([f"Plot {i+1}" for i in range(len(titles), n_plots)])
        
        # Create subplots
        dashboard = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=titles[:n_plots],
            vertical_spacing=kwargs.get('vertical_spacing', 0.15),
            horizontal_spacing=kwargs.get('horizontal_spacing', 0.15),
            specs=[[{"type": "xy"}] * cols for _ in range(rows)]
        )
        
        # Add each figure to the dashboard
        for idx, fig in enumerate(figures):
            row = (idx // cols) + 1
            col = (idx % cols) + 1
            
            # Add all traces from the figure
            for trace in fig.data:
                dashboard.add_trace(
                    trace,
                    row=row,
                    col=col
                )
            
            # Update axes for this subplot
            xaxis = f'xaxis{idx+1}' if idx > 0 else 'xaxis'
            yaxis = f'yaxis{idx+1}' if idx > 0 else 'yaxis'
            
            # Get original axis titles if available
            if fig.layout.xaxis.title.text:
                dashboard.layout[xaxis].title = fig.layout.xaxis.title.text
            if fig.layout.yaxis.title.text:
                dashboard.layout[yaxis].title = fig.layout.yaxis.title.text
        
        # Update overall layout
        dashboard.update_layout(
            height=kwargs.get('height', 400 * rows),
            showlegend=kwargs.get('showlegend', False),
            title_text=kwargs.get('title', 'Dashboard'),
            title_font_size=20,
            font=dict(size=10),
            margin=dict(l=60, r=60, t=100, b=60)
        )
        
        return dashboard
