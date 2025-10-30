import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import json
import plotly.graph_objects as go
import os
# Import custom modules
from config import Config
from agents.data_processor import DataProcessor
from agents.query_processor import QueryProcessor
from agents.analysis_agent import AnalysisAgent
from agents.viz_agent import VizAgent

# Page configuration
st.set_page_config(
    page_title="Agentic Data Analyst",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ensure upload directory exists
Config.ensure_upload_dir()

# Initialize session state - Fixed initialization
@st.cache_resource
def get_query_processor():
    return QueryProcessor()

def initialize_session_state():
    """Initialize session state variables"""
    if 'data_processor' not in st.session_state:
        st.session_state.data_processor = DataProcessor()
    if 'query_processor' not in st.session_state:
        st.session_state.query_processor = get_query_processor()
    if 'analysis_history' not in st.session_state:
        st.session_state.analysis_history = []

def main():
    """Main application function"""
    # Initialize session state
    initialize_session_state()
    
    st.title("🤖 Agentic AI Data Analyst & Visualization Agent")
    st.markdown("Upload your dataset and ask questions in natural language!")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        if not Config.GROQ_API_KEY:
            st.error("⚠️ Groq API Key not found!")
            st.markdown("Please set your GROQ_API_KEY in a .env file or environment variable")
        else:
            if st.session_state.query_processor.client:
                st.success("✅ Groq API Connected")
            else:
                st.warning("⚠️ Groq API connection failed - using fallback mode")
        
        st.markdown("---")
        st.header("📁 Data Upload")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=Config.ALLOWED_EXTENSIONS,
            help="Supported formats: CSV, Excel, JSON, Parquet"
        )
        
        if uploaded_file is not None:
            if st.button("Load Data", type="primary"):
                with st.spinner("Loading data..."):
                    success, message = st.session_state.data_processor.load_file(uploaded_file)
                    if success:
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)
    
    # Main content area
    if st.session_state.data_processor and st.session_state.data_processor.data is not None:
        # Add Smart Analysis tab
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Data Overview",
            "🤖 **Smart Analysis**",  # NEW TAB
            "🔍 Natural Language Query",
            "📈 Custom Visualizations",
            "📜 Analysis History"
        ])
        
        with tab1:
            show_data_overview()
        
        with tab2:
            show_smart_analysis()  # NEW FUNCTION
        
        with tab3:
            show_analysis_interface()
        
        with tab4:
            show_visualization_interface()
        
        with tab5:
            show_analysis_history()

    
    else:
        st.info("👆 Please upload a dataset to begin analysis")
        
        # Show sample data info
        st.markdown("### 📝 Supported File Formats")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Supported formats:**
            - CSV (.csv)
            - Excel (.xlsx, .xls)  
            - JSON (.json)
            - Parquet (.parquet)
            """)
        
        with col2:
            st.markdown("""
            **Example queries you can ask:**
            - "What is the average sales by region?"
            - "Show me the correlation between price and quantity"
            - "Create a trend chart for revenue over time"
            - "Compare product performance across categories"
            """)

def show_data_overview():
    """Display data overview and basic statistics"""
    data = st.session_state.data_processor.data
    metadata = st.session_state.data_processor.metadata
    
    st.header("📊 Dataset Overview")
    
    # Basic info
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Rows", f"{metadata['shape'][0]:,}")
    with col2:
        st.metric("Columns", metadata['shape'][1])
    with col3:
        st.metric("Missing Values", sum(metadata['null_counts'].values()))
    with col4:
        st.metric("Duplicates", metadata['duplicate_rows'])
    
    # Data preview
    st.subheader("🔍 Data Preview")
    st.dataframe(data.head(100), use_container_width=True)
    
    # Column information
    st.subheader("📋 Column Information")
    col_info = []
    for col, dtype in metadata['dtypes'].items():
        col_info.append({
            'Column': col,
            'Type': str(dtype),
            'Non-Null Count': metadata['shape'][0] - metadata['null_counts'][col],
            'Null Count': metadata['null_counts'][col],
            'Unique Values': data[col].nunique()
        })
    
    st.dataframe(pd.DataFrame(col_info), use_container_width=True)
    
    # Data quality insights
    st.subheader("🎯 Data Quality Insights")
    
    # Missing data visualization
    if sum(metadata['null_counts'].values()) > 0:
        missing_data = pd.DataFrame(list(metadata['null_counts'].items()), 
                                   columns=['Column', 'Missing_Count'])
        missing_data = missing_data[missing_data['Missing_Count'] > 0]
        
        if len(missing_data) > 0:
            viz_agent = VizAgent()
            fig = viz_agent.create_visualization(
                missing_data, 
                'bar', 
                x='Column', 
                y='Missing_Count',
                title='Missing Values by Column'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Data types distribution
    dtype_counts = pd.DataFrame(list(metadata['dtypes'].values()), columns=['Data_Type'])
    dtype_summary = dtype_counts['Data_Type'].value_counts().reset_index()
    dtype_summary.columns = ['Data_Type', 'Count']
    
    if len(dtype_summary) > 0:
        viz_agent = VizAgent()
        fig = viz_agent.create_visualization(
            dtype_summary,
            'pie',
            names='Data_Type',
            values='Count',
            title='Data Types Distribution'
        )
        st.plotly_chart(fig, use_container_width=True)
def show_analysis_interface():
    """Display enhanced analysis interface with natural language queries"""
    st.header("🔍 Natural Language Analysis")
    st.markdown("Ask questions about your data in plain English, and I'll provide clear insights!")
    
    data = st.session_state.data_processor.data
    columns = list(data.columns)
    numeric_columns = st.session_state.data_processor.get_numeric_columns()
    categorical_columns = st.session_state.data_processor.get_categorical_columns()
    
    # Initialize query in session state if not exists
    if 'current_query' not in st.session_state:
        st.session_state.current_query = ""
    
    # Enhanced query input with examples
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.markdown("**💡 Example Questions:**")
        example_queries = [
            "Summarize the main trends",
            "What are the top performers?", 
            "Show me correlations",
            "Compare categories"
        ]
        
        for i, example in enumerate(example_queries):
            if st.button(f"📝 {example}", key=f"example_{i}"):
                st.session_state.current_query = example
                st.rerun()
    
    with col1:
        query = st.text_input(
            "💭 Ask me anything about your data:",
            value=st.session_state.current_query,
            placeholder="e.g., What are the key patterns in sales data?",
            help="Ask in natural language - I'll understand and explain in simple terms!"
        )
        # Update session state when text input changes
        if query != st.session_state.current_query:
            st.session_state.current_query = query

    # Enhanced quick suggestions
    st.markdown("---")
    st.subheader("⚡ Quick Analysis Options")
    
    quick_col1, quick_col2, quick_col3, quick_col4 = st.columns(4)
    
    with quick_col1:
        if st.button("📊 **Data Summary**", use_container_width=True):
            st.session_state.current_query = "Give me a comprehensive summary of this dataset"
            st.rerun()
    
    with quick_col2:
        if st.button("🔗 **Find Relationships**", use_container_width=True) and len(numeric_columns) >= 2:
            st.session_state.current_query = f"What is the relationship between {numeric_columns[0]} and other variables?"
            st.rerun()
    
    with quick_col3:
        if st.button("📈 **Show Trends**", use_container_width=True) and len(numeric_columns) >= 1:
            st.session_state.current_query = f"Show me the distribution and trends in {numeric_columns[0]}"
            st.rerun()
    
    with quick_col4:
        if st.button("🏆 **Top Insights**", use_container_width=True):
            st.session_state.current_query = "What are the most important insights from this data?"
            st.rerun()
    
    # Use the query from session state
    query = st.session_state.current_query
    
    if st.button("🚀 **Analyze & Explain**", type="primary", use_container_width=True) and query:
        with st.spinner("🤔 Analyzing your data and preparing insights..."):
            try:
                # Process query
                data_summary = {
                    'columns': columns,
                    'numeric_columns': numeric_columns,
                    'categorical_columns': categorical_columns,
                    'shape': data.shape
                }
                
                query_result = st.session_state.query_processor.process_query(
                    query, columns, data_summary
                )
                
                # Initialize analysis components
                analysis_agent = AnalysisAgent(data)
                viz_agent = VizAgent()
                
                # **ENHANCED RESPONSE FORMATTING**
                st.markdown("---")
                
                # Main response header
                st.markdown(f"""
                ## 🎯 Analysis Results for: *"{query}"*
                
                Let me break this down for you in simple terms:
                """)
                
                results = {}
                insights_text = []
                
                # Execute analysis based on intent with enhanced explanations
                if query_result['intent'] == 'descriptive':
                    if query_result['analysis_type'] == 'groupby' and query_result.get('target_columns') and query_result.get('groupby_columns'):
                        try:
                            result = analysis_agent.groupby_analysis(
                                query_result['target_columns'][0],
                                query_result['groupby_columns'][0],
                                query_result.get('aggregation', 'mean')
                            )
                            
                            # **ENHANCED EXPLANATION**
                            target_col = query_result['target_columns'][0]
                            group_col = query_result['groupby_columns'][0]
                            agg_func = query_result.get('aggregation', 'mean')
                            
                            st.markdown(f"""
                            ### 📊 **Key Finding**: {agg_func.title()} {target_col.replace('_', ' ').title()} by {group_col.replace('_', ' ').title()}
                            
                            Here's what I found when I looked at how **{target_col.replace('_', ' ')}** varies across different **{group_col.replace('_', ' ')}** categories:
                            """)
                            
                            # Show top 3 and bottom 3 with explanations
                            top_3 = result.head(3)
                            bottom_3 = result.tail(3)
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("#### 🏆 **Top Performers:**")
                                for idx, row in top_3.iterrows():
                                    st.markdown(f"• **{row[group_col]}**: {row[target_col]:.2f}")
                                
                            with col2:
                                st.markdown("#### 📉 **Lowest Performers:**")
                                for idx, row in bottom_3.iterrows():
                                    st.markdown(f"• **{row[group_col]}**: {row[target_col]:.2f}")
                            
                            # Detailed table
                            with st.expander("📋 View Complete Results", expanded=False):
                                st.dataframe(result, use_container_width=True)
                            
                            # Enhanced visualization
                            if len(result) <= 20:
                                fig = viz_agent.create_visualization(
                                    result,
                                    'bar',
                                    x=group_col,
                                    y=target_col,
                                    title=f"{agg_func.title()} {target_col.replace('_', ' ').title()} by {group_col.replace('_', ' ').title()}"
                                )
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Insight explanation
                                max_val = result[target_col].max()
                                min_val = result[target_col].min()
                                max_category = result.loc[result[target_col].idxmax(), group_col]
                                min_category = result.loc[result[target_col].idxmin(), group_col]
                                
                                insights_text.extend([
                                    f"**{max_category}** has the highest {target_col.replace('_', ' ')} at {max_val:.2f}",
                                    f"**{min_category}** has the lowest {target_col.replace('_', ' ')} at {min_val:.2f}",
                                    f"The difference between highest and lowest is {max_val - min_val:.2f}"
                                ])
                            
                            results['groupby_result'] = result.to_dict()
                            
                        except Exception as e:
                            st.error(f"❌ I encountered an issue: {str(e)}")
                    
                    else:
                        # General descriptive analysis with enhanced explanation
                        target_cols = query_result.get('target_columns', numeric_columns[:3])
                        if target_cols:
                            result = analysis_agent.descriptive_analysis(target_cols)
                            
                            st.markdown(f"""
                            ### 📈 **Statistical Summary**
                            
                            I've analyzed the key statistics for your numeric columns. Here's what stands out:
                            """)
                            
                            if result:
                                # Create user-friendly summary
                                for col, stats in result.items():
                                    if isinstance(stats, dict) and 'mean' in stats:
                                        st.markdown(f"""
                                        **📊 {col.replace('_', ' ').title()}:**
                                        - Average value: **{stats['mean']:.2f}**
                                        - Ranges from **{stats['min']:.2f}** to **{stats['max']:.2f}**
                                        - Most values fall between **{stats['q25']:.2f}** and **{stats['q75']:.2f}**
                                        """)
                                        
                                        insights_text.append(f"The average {col.replace('_', ' ')} is {stats['mean']:.2f}")
                                
                                # Detailed stats in expander
                                with st.expander("📊 Detailed Statistics", expanded=False):
                                    df_stats = pd.DataFrame(result).T
                                    st.dataframe(df_stats, use_container_width=True)
                                
                                results['descriptive_stats'] = result
                
                elif query_result['intent'] == 'correlation':
                    try:
                        if len(numeric_columns) >= 2:
                            corr_matrix = analysis_agent.correlation_analysis()
                            
                            st.markdown(f"""
                            ### 🔗 **Relationship Analysis**
                            
                            I've examined how your numeric variables relate to each other. Here's what I discovered:
                            """)
                            
                            # Find strongest correlations
                            corr_pairs = []
                            for i in range(len(corr_matrix.columns)):
                                for j in range(i+1, len(corr_matrix.columns)):
                                    col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                                    corr_val = corr_matrix.iloc[i, j]
                                    if abs(corr_val) > 0.3:  # Only significant correlations
                                        corr_pairs.append((col1, col2, corr_val))
                            
                            corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
                            
                            if corr_pairs:
                                st.markdown("#### 🔍 **Strongest Relationships Found:**")
                                for col1, col2, corr_val in corr_pairs[:5]:
                                    direction = "positively" if corr_val > 0 else "negatively"
                                    strength = "strongly" if abs(corr_val) > 0.7 else "moderately"
                                    st.markdown(f"• **{col1.replace('_', ' ').title()}** and **{col2.replace('_', ' ').title()}** are {strength} {direction} related (correlation: {corr_val:.2f})")
                            else:
                                st.markdown("• No strong relationships found between variables (all correlations < 0.3)")
                            
                            # Correlation heatmap
                            fig = viz_agent.create_visualization(
                                corr_matrix, 'correlation',
                                title='Variable Relationships Heatmap'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            with st.expander("📋 Correlation Matrix Table", expanded=False):
                                st.dataframe(corr_matrix, use_container_width=True)
                            
                            results['correlation_matrix'] = corr_matrix.to_dict()
                        else:
                            st.warning("⚠️ I need at least 2 numeric columns to analyze relationships.")
                    
                    except Exception as e:
                        st.error(f"❌ Error in correlation analysis: {str(e)}")
                
                elif query_result['intent'] == 'visualization':
                    st.markdown(f"""
                    ### 📈 **Visual Analysis**
                    
                    I've created a visualization to help you see the patterns in your data:
                    """)
                    
                    if query_result.get('visualization_type') and query_result.get('target_columns'):
                        viz_type = query_result['visualization_type']
                        target_cols = query_result['target_columns']
                        
                        try:
                            fig = None
                            
                            if viz_type == 'histogram' and len(target_cols) >= 1:
                                col_name = target_cols[0]
                                fig = viz_agent.create_visualization(
                                    data, 'histogram', x=col_name,
                                    title=f'Distribution of {col_name.replace("_", " ").title()}'
                                )
                                
                                # Add distribution insights
                                mean_val = data[col_name].mean()
                                median_val = data[col_name].median()
                                std_val = data[col_name].std()
                                
                                st.markdown(f"""
                                **📊 Distribution Insights:**
                                - The average {col_name.replace('_', ' ')} is **{mean_val:.2f}**
                                - Half the values are below **{median_val:.2f}** (median)
                                - Most values fall within **{mean_val - std_val:.2f}** to **{mean_val + std_val:.2f}**
                                """)
                            
                            elif viz_type in ['bar', 'line'] and len(target_cols) >= 2:
                                fig = viz_agent.create_visualization(
                                    data, viz_type, 
                                    x=target_cols[1], y=target_cols[0],
                                    title=f'{target_cols[0].replace("_", " ").title()} by {target_cols[1].replace("_", " ").title()}'
                                )
                            
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                                results['visualization_created'] = True
                        
                        except Exception as e:
                            st.error(f"❌ Error creating visualization: {str(e)}")
                
                # **ENHANCED INSIGHTS SECTION**
                st.markdown("---")
                st.markdown("### 💡 **Key Takeaways**")
                
                if insights_text:
                    for insight in insights_text:
                        st.markdown(f"• {insight}")
                
                # Generate AI insights
                ai_insights = st.session_state.query_processor.generate_insights(
                    {'query_result': query_result, 'results': results}, query
                )
                
                if ai_insights and ai_insights != "Analysis completed successfully. Please review the results and visualizations above for insights.":
                    st.markdown("### 🤖 **Additional AI Insights**")
                    st.markdown(ai_insights)
                
                # **ACTION SUGGESTIONS**
                st.markdown("### 🎯 **What You Can Do Next**")
                
                suggestions = []
                if query_result['intent'] == 'descriptive':
                    suggestions.extend([
                        "Try asking about relationships: 'How do these variables relate to each other?'",
                        "Explore trends: 'Show me trends over time'",
                        "Compare categories: 'Compare performance across groups'"
                    ])
                elif query_result['intent'] == 'correlation':
                    suggestions.extend([
                        "Investigate strong correlations further",
                        "Look for causal relationships behind correlations",
                        "Consider other factors that might influence these relationships"
                    ])
                elif query_result['intent'] == 'visualization':
                    suggestions.extend([
                        "Try different chart types to see other patterns",
                        "Filter the data to focus on specific segments",
                        "Look for outliers or unusual data points"
                    ])
                
                for suggestion in suggestions:
                    st.markdown(f"• {suggestion}")
                
                # Save to history with enhanced format
                st.session_state.analysis_history.append({
                    'query': query,
                    'timestamp': pd.Timestamp.now(),
                    'query_result': query_result,
                    'insights': ai_insights,
                    'results': results,
                    'user_friendly_summary': f"Analyzed {query_result['intent']} query about {', '.join(query_result.get('target_columns', ['data']))}"
                })
                
                st.success("✅ Analysis completed! Feel free to ask another question.")
                
                # Clear the query after successful analysis
                st.session_state.current_query = ""
                
            except Exception as e:
                st.error(f"❌ I encountered an unexpected issue: {str(e)}")
                st.info("💡 Try rephrasing your question or use one of the quick suggestions above.")
def show_visualization_interface():
    """Display visualization creation interface"""
    st.header("📈 Custom Visualizations")
    
    data = st.session_state.data_processor.data
    columns = list(data.columns)
    numeric_columns = st.session_state.data_processor.get_numeric_columns()
    categorical_columns = st.session_state.data_processor.get_categorical_columns()
    
    viz_agent = VizAgent()
    
    # Initialize session state for quick analysis
    if 'show_quick_analysis' not in st.session_state:
        st.session_state.show_quick_analysis = None
    
    # Quick analysis buttons at the top
    st.subheader("⚡ Quick Analysis")
    
    quick_col1, quick_col2, quick_col3, quick_col4 = st.columns(4)
    
    with quick_col1:
        if st.button("📊 Correlation Matrix", use_container_width=True) and len(numeric_columns) >= 2:
            st.session_state.show_quick_analysis = 'correlation'
            st.rerun()
    
    with quick_col2:
        if st.button("📈 Distribution Overview", use_container_width=True) and len(numeric_columns) >= 1:
            st.session_state.show_quick_analysis = 'distribution'
            st.rerun()
    
    with quick_col3:
        if st.button("🎯 Summary Statistics", use_container_width=True) and len(numeric_columns) >= 1:
            st.session_state.show_quick_analysis = 'statistics'
            st.rerun()
    
    with quick_col4:
        if st.button("🔍 Data Quality Check", use_container_width=True):
            st.session_state.show_quick_analysis = 'quality'
            st.rerun()
    
    # Display quick analysis in full screen if selected
    if st.session_state.show_quick_analysis:
        st.markdown("---")
        
        # Add close button
        col_close1, col_close2 = st.columns([6, 1])
        with col_close2:
            if st.button("❌ Close", key="close_quick_analysis"):
                st.session_state.show_quick_analysis = None
                st.rerun()
        
        if st.session_state.show_quick_analysis == 'correlation':
            st.subheader("📊 Correlation Matrix Analysis")
            st.markdown("Understanding relationships between your numeric variables:")
            
            try:
                analysis_agent = AnalysisAgent(data)
                corr_matrix = analysis_agent.correlation_analysis()
                
                # Create visualization
                fig = viz_agent.create_visualization(
                    corr_matrix, 
                    'correlation',
                    title='Variable Correlation Matrix',
                    height=700
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Find strongest correlations
                st.markdown("### 🔍 Strongest Correlations")
                corr_pairs = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                        corr_val = corr_matrix.iloc[i, j]
                        if abs(corr_val) > 0.3:
                            corr_pairs.append((col1, col2, corr_val))
                
                corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
                
                if corr_pairs:
                    for col1, col2, corr_val in corr_pairs[:5]:
                        direction = "positive" if corr_val > 0 else "negative"
                        strength = "strong" if abs(corr_val) > 0.7 else "moderate"
                        st.markdown(f"• **{col1}** ↔ **{col2}**: {strength} {direction} correlation ({corr_val:.3f})")
                else:
                    st.info("No strong correlations found (all < 0.3)")
                
                # Show full correlation table
                with st.expander("📋 View Full Correlation Table"):
                    st.dataframe(corr_matrix, use_container_width=True)
                    
            except Exception as e:
                st.error(f"Error creating correlation matrix: {str(e)}")
        
        elif st.session_state.show_quick_analysis == 'distribution':
            st.subheader("📈 Distribution Overview")
            st.markdown("Visualizing the distribution of your numeric variables:")
            
            try:
                # Create histograms for numeric columns
                num_cols_to_show = min(len(numeric_columns), 6)
                figs = []
                
                for col in numeric_columns[:num_cols_to_show]:
                    fig = viz_agent.create_visualization(
                        data, 
                        'histogram', 
                        x=col, 
                        title=f'Distribution of {col.replace("_", " ").title()}'
                    )
                    figs.append(fig)
                
                if figs:
                    # Determine grid layout
                    if len(figs) <= 2:
                        rows, cols = 1, 2
                    elif len(figs) <= 4:
                        rows, cols = 2, 2
                    else:
                        rows, cols = 3, 2
                    
                    dashboard = viz_agent.create_dashboard(
                        figs, 
                        [f'Distribution of {col.replace("_", " ").title()}' for col in numeric_columns[:num_cols_to_show]],
                        rows=rows,
                        cols=cols,
                        title='Distribution Dashboard',
                        height=300 * rows
                    )
                    st.plotly_chart(dashboard, use_container_width=True)
                    
                    # Add statistical insights
                    st.markdown("### 📊 Key Statistics")
                    stats_cols = st.columns(min(3, len(numeric_columns)))
                    
                    for idx, col in enumerate(numeric_columns[:num_cols_to_show]):
                        with stats_cols[idx % 3]:
                            mean_val = data[col].mean()
                            median_val = data[col].median()
                            std_val = data[col].std()
                            
                            st.metric(
                                label=col.replace('_', ' ').title(),
                                value=f"{mean_val:.2f}",
                                delta=f"σ: {std_val:.2f}"
                            )
                            st.caption(f"Median: {median_val:.2f}")
                            
            except Exception as e:
                st.error(f"Error creating distribution overview: {str(e)}")
        
        elif st.session_state.show_quick_analysis == 'statistics':
            st.subheader("🎯 Summary Statistics")
            st.markdown("Comprehensive statistical summary of your numeric data:")
            
            try:
                analysis_agent = AnalysisAgent(data)
                stats = analysis_agent.descriptive_analysis(numeric_columns)
                stats_df = pd.DataFrame(stats).T
                
                # Display stats table
                st.dataframe(stats_df.style.format("{:.2f}"), use_container_width=True)
                
                # Add visual summary
                st.markdown("### 📊 Visual Summary")
                
                # Create box plots for comparison
                if len(numeric_columns) <= 6:
                    # Normalize data for comparison
                    from sklearn.preprocessing import StandardScaler
                    scaler = StandardScaler()
                    normalized_data = data[numeric_columns].copy()
                    normalized_data[numeric_columns] = scaler.fit_transform(data[numeric_columns])
                    
                    # Melt for box plot
                    melted_data = normalized_data.melt(var_name='Variable', value_name='Normalized Value')
                    
                    fig = viz_agent.create_visualization(
                        melted_data,
                        'box',
                        x='Variable',
                        y='Normalized Value',
                        title='Normalized Distribution Comparison',
                        height=500
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption("Note: Data is normalized (z-score) for comparison across variables")
                
                # Download button
                csv = stats_df.to_csv()
                st.download_button(
                    label="📥 Download Statistics (CSV)",
                    data=csv,
                    file_name="summary_statistics.csv",
                    mime="text/csv"
                )
                
            except Exception as e:
                st.error(f"Error calculating statistics: {str(e)}")
        
        elif st.session_state.show_quick_analysis == 'quality':
            st.subheader("🔍 Data Quality Check")
            st.markdown("Comprehensive data quality assessment:")
            
            try:
                # Create quality report
                quality_info = []
                for col in columns:
                    null_count = data[col].isnull().sum()
                    null_pct = (null_count / len(data)) * 100
                    unique_count = data[col].nunique()
                    
                    quality_info.append({
                        'Column': col,
                        'Type': str(data[col].dtype),
                        'Missing': null_count,
                        'Missing %': round(null_pct, 2),
                        'Unique Values': unique_count,
                        'Uniqueness %': round((unique_count / len(data)) * 100, 2),
                        'Most Frequent': str(data[col].mode().iloc[0]) if not data[col].mode().empty else 'N/A'
                    })
                
                quality_df = pd.DataFrame(quality_info)
                
                # Display quality metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    complete_cols = len(quality_df[quality_df['Missing'] == 0])
                    st.metric("Complete Columns", f"{complete_cols}/{len(columns)}")
                
                with col2:
                    avg_missing = quality_df['Missing %'].mean()
                    st.metric("Avg Missing %", f"{avg_missing:.2f}%")
                
                with col3:
                    duplicate_rows = data.duplicated().sum()
                    st.metric("Duplicate Rows", duplicate_rows)
                
                with col4:
                    total_rows = len(data)
                    st.metric("Total Rows", total_rows)
                
                # Display quality table
                st.markdown("### 📋 Column Quality Report")
                
                # Color code the table
                def highlight_quality(row):
                    if row['Missing %'] > 50:
                        return ['background-color: #ffcccc'] * len(row)
                    elif row['Missing %'] > 20:
                        return ['background-color: #ffffcc'] * len(row)
                    else:
                        return ['background-color: #ccffcc'] * len(row)
                
                styled_df = quality_df.style.apply(highlight_quality, axis=1)
                st.dataframe(styled_df, use_container_width=True)
                
                # Missing data visualization
                if quality_df['Missing %'].max() > 0:
                    st.markdown("### 📉 Missing Data Visualization")
                    missing_data = quality_df[quality_df['Missing %'] > 0].sort_values('Missing %', ascending=False)
                    
                    if not missing_data.empty:
                        fig = viz_agent.create_visualization(
                            missing_data,
                            'bar',
                            x='Column',
                            y='Missing %',
                            title='Missing Data by Column',
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                # Download button
                csv = quality_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Quality Report (CSV)",
                    data=csv,
                    file_name="data_quality_report.csv",
                    mime="text/csv"
                )
                
            except Exception as e:
                st.error(f"Error performing quality check: {str(e)}")
        
        st.markdown("---")
        return  # Exit early to show only quick analysis
    
    # Chart Builder (only shown when no quick analysis is active)
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("🎨 Chart Builder")
        
        chart_type = st.selectbox(
            "Chart Type",
            ['bar', 'line', 'scatter', 'histogram', 'box', 'violin', 'pie', 'heatmap', 'area']
        )
        
        if chart_type in ['bar', 'line', 'scatter', 'area']:
            x_col = st.selectbox("X-axis", columns)
            y_col = st.selectbox("Y-axis", [col for col in columns if col != x_col])
            color_col = st.selectbox("Color by (optional)", [None] + [col for col in columns if col not in [x_col, y_col]])
            
        elif chart_type == 'histogram':
            x_col = st.selectbox("Column", numeric_columns)
            y_col = None
            color_col = st.selectbox("Color by (optional)", [None] + categorical_columns)
            
        elif chart_type in ['box', 'violin']:
            x_col = st.selectbox("Category (optional)", [None] + categorical_columns)
            y_col = st.selectbox("Values", numeric_columns)
            color_col = None
            
        elif chart_type == 'pie':
            x_col = st.selectbox("Categories", categorical_columns)
            y_col = st.selectbox("Values", numeric_columns)
            color_col = None
            
        elif chart_type == 'heatmap':
            x_col = None
            y_col = None
            color_col = None
        
        # Chart customization
        st.subheader("🎨 Customization")
        chart_title = st.text_input("Chart Title", value=f"{chart_type.title()} Chart")
        chart_height = st.slider("Chart Height", 300, 800, 500)
        
        if st.button("Create Visualization", type="primary"):
            try:
                if chart_type == 'pie':
                    # Aggregate data for pie chart
                    pie_data = data.groupby(x_col)[y_col].sum().reset_index()
                    fig = viz_agent.create_visualization(
                        pie_data, chart_type, 
                        names=x_col, values=y_col,
                        title=chart_title,
                        height=chart_height
                    )
                elif chart_type == 'heatmap':
                    fig = viz_agent.create_visualization(
                        data, 'correlation',
                        title=chart_title,
                        height=chart_height
                    )
                else:
                    fig = viz_agent.create_visualization(
                        data, chart_type,
                        x=x_col, y=y_col, color=color_col,
                        title=chart_title,
                        height=chart_height
                    )
                
                with col2:
                    st.subheader("📊 Visualization")
                    st.plotly_chart(fig, use_container_width=True)
                    
            except Exception as e:
                st.error(f"Error creating visualization: {str(e)}")

def show_analysis_history():
    """Display analysis history"""
    st.header("📋 Analysis History")
    
    if not st.session_state.analysis_history:
        st.info("No analysis history yet. Start by asking questions about your data!")
        return
    
    # Controls
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🗑️ Clear History"):
            st.session_state.analysis_history = []
            st.rerun()
    
    # Display history
    for i, entry in enumerate(reversed(st.session_state.analysis_history)):
        with st.expander(f"📝 Query {len(st.session_state.analysis_history) - i}: {entry['query'][:60]}...", expanded=False):
            
            # Query details
            st.markdown(f"**🕐 Time:** {entry['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
            st.markdown(f"**❓ Query:** {entry['query']}")
            st.markdown(f"**🎯 Intent:** {entry['query_result'].get('intent', 'N/A')}")
            st.markdown(f"**🔍 Analysis Type:** {entry['query_result'].get('analysis_type', 'N/A')}")
            
            # Insights
            st.markdown("**💡 Insights:**")
            st.markdown(entry['insights'])
            
            # Results summary if available
            if 'results' in entry and entry['results']:
                st.markdown("**📊 Results Summary:**")
                for key, value in entry['results'].items():
                    if isinstance(value, dict) and len(value) < 10:  # Show small dictionaries
                        st.json(value)
                    elif isinstance(value, bool):
                        st.write(f"- {key}: {'✅ Yes' if value else '❌ No'}")
                    else:
                        st.write(f"- {key}: Available")

# Additional utility functions
def download_results():
    """Function to download analysis results"""
    if st.session_state.analysis_history:
        history_df = pd.DataFrame([
            {
                'Query': entry['query'],
                'Timestamp': entry['timestamp'],
                'Intent': entry['query_result'].get('intent', 'N/A'),
                'Analysis_Type': entry['query_result'].get('analysis_type', 'N/A'),
                'Insights': entry['insights']
            }
            for entry in st.session_state.analysis_history
        ])
        
        csv = history_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Analysis History",
            data=csv,
            file_name=f"analysis_history_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
        
def show_smart_analysis():
    """Display AI-powered smart analysis suggestions"""
    st.header("🤖 Smart Analysis Recommendations")
    st.markdown("""
    Let AI analyze your data and suggest the **3 most valuable insights** to explore.
    Each suggestion comes with visualizations and explanations.
    """)
    
    data = st.session_state.data_processor.data
    
    # Initialize Smart Analysis Agent
    from agents.smart_analysis_agent import SmartAnalysisAgent
    
    if st.button("🔍 **Analyze My Data & Get Recommendations**", type="primary", use_container_width=True):
        with st.spinner("🤔 AI is analyzing your data structure and patterns..."):
            try:
                smart_agent = SmartAnalysisAgent(
                    data=data,
                    groq_api_key=os.getenv('GROQ_API_KEY')
                )
                
                suggestions = smart_agent.generate_smart_suggestions()
                
                st.success("✅ Analysis complete! Here are the top 3 insights to explore:")
                
                # Display each suggestion
                for idx, suggestion in enumerate(suggestions, 1):
                    st.markdown("---")
                    st.markdown(f"## 💡 Insight #{idx}: {suggestion['title']}")
                    
                    # Show description and explanation
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown(f"**{suggestion['description']}**")
                        
                        # Get AI explanation
                        with st.expander("🧠 Why This Matters", expanded=True):
                            ai_explanation = smart_agent.generate_ai_explanation(suggestion)
                            st.markdown(ai_explanation)
                    
                    with col2:
                        st.info(f"**Analysis Type:** {suggestion['type'].title()}")
                        st.info(f"**Columns:** {', '.join(suggestion['columns'])}")
                    
                    # Execute the analysis and create visualizations
                    st.markdown("### 📊 Visualization & Results")
                    
                    analysis_agent = AnalysisAgent(data)
                    viz_agent = VizAgent()
                    
                    if suggestion['type'] == 'distribution':
                        # Create distribution dashboard
                        figs = []
                        stats_data = []
                        
                        for col in suggestion['columns']:
                            fig = viz_agent.create_visualization(
                                data, 'histogram', x=col,
                                title=f'Distribution of {col.replace("_", " ").title()}'
                            )
                            figs.append(fig)
                            
                            # Add statistics
                            stats_data.append({
                                'Column': col,
                                'Mean': data[col].mean(),
                                'Median': data[col].median(),
                                'Std Dev': data[col].std(),
                                'Min': data[col].min(),
                                'Max': data[col].max()
                            })
                        
                        if figs:
                            dashboard = viz_agent.create_dashboard(
                                figs,
                                [f'Distribution of {col}' for col in suggestion['columns']],
                                rows=len(figs) if len(figs) <= 2 else 2,
                                cols=2 if len(figs) > 1 else 1,
                                height=400 * (len(figs) if len(figs) <= 2 else 2)
                            )
                            st.plotly_chart(dashboard, use_container_width=True)
                        
                        # Show statistics table
                        with st.expander("📈 Detailed Statistics"):
                            stats_df = pd.DataFrame(stats_data)
                            st.dataframe(stats_df.style.format({
                                'Mean': '{:.2f}',
                                'Median': '{:.2f}',
                                'Std Dev': '{:.2f}',
                                'Min': '{:.2f}',
                                'Max': '{:.2f}'
                            }), use_container_width=True)
                    
                    elif suggestion['type'] == 'correlation':
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Scatter plot of top correlated pair
                            if len(suggestion['columns']) >= 2:
                                fig_scatter = viz_agent.create_visualization(
                                    data, 'scatter',
                                    x=suggestion['columns'][0],
                                    y=suggestion['columns'][1],
                                    title=f'{suggestion["columns"][1]} vs {suggestion["columns"][0]}',
                                    trendline=True
                                )
                                st.plotly_chart(fig_scatter, use_container_width=True)
                        
                        with col2:
                            # Correlation matrix
                            corr_matrix = analysis_agent.correlation_analysis()
                            fig_corr = viz_agent.create_visualization(
                                corr_matrix, 'correlation',
                                title='Correlation Matrix'
                            )
                            st.plotly_chart(fig_corr, use_container_width=True)
                        
                        # Show top correlations
                        with st.expander("🔍 Top Correlations"):
                            corr_pairs = []
                            for i in range(len(corr_matrix.columns)):
                                for j in range(i+1, len(corr_matrix.columns)):
                                    col1_name, col2_name = corr_matrix.columns[i], corr_matrix.columns[j]
                                    corr_val = corr_matrix.iloc[i, j]
                                    if abs(corr_val) > 0.3:
                                        corr_pairs.append({
                                            'Variable 1': col1_name,
                                            'Variable 2': col2_name,
                                            'Correlation': corr_val,
                                            'Strength': 'Strong' if abs(corr_val) > 0.7 else 'Moderate'
                                        })
                            
                            if corr_pairs:
                                corr_df = pd.DataFrame(corr_pairs).sort_values('Correlation', key=abs, ascending=False)
                                st.dataframe(corr_df.style.format({'Correlation': '{:.3f}'}), use_container_width=True)
                    
                    elif suggestion['type'] == 'groupby':
                        num_col, cat_col = suggestion['columns']
                        
                        # Group by analysis
                        result = analysis_agent.groupby_analysis(num_col, cat_col, 'mean')
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Bar chart
                            if len(result) <= 15:
                                fig_bar = viz_agent.create_visualization(
                                    result, 'bar',
                                    x=cat_col, y=num_col,
                                    title=f'Average {num_col} by {cat_col}'
                                )
                                st.plotly_chart(fig_bar, use_container_width=True)
                        
                        with col2:
                            # Box plot
                            fig_box = viz_agent.create_visualization(
                                data, 'box',
                                x=cat_col, y=num_col,
                                title=f'{num_col} Distribution by {cat_col}'
                            )
                            st.plotly_chart(fig_box, use_container_width=True)
                        
                        # Show summary statistics
                        with st.expander("📊 Category Statistics"):
                            st.dataframe(result, use_container_width=True)
                    
                    elif suggestion['type'] == 'timeseries':
                        num_col, date_col = suggestion['columns']
                        
                        # Sort by date
                        time_data = data[[date_col, num_col]].sort_values(date_col)
                        
                        fig_line = viz_agent.create_visualization(
                            time_data, 'line',
                            x=date_col, y=num_col,
                            title=f'{num_col} Over Time'
                        )
                        st.plotly_chart(fig_line, use_container_width=True)
                        
                        # Show trend statistics
                        with st.expander("📈 Trend Analysis"):
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Starting Value", f"{time_data[num_col].iloc[0]:.2f}")
                            with col2:
                                st.metric("Current Value", f"{time_data[num_col].iloc[-1]:.2f}")
                            with col3:
                                change = time_data[num_col].iloc[-1] - time_data[num_col].iloc[0]
                                st.metric("Change", f"{change:.2f}", delta=f"{change:.2f}")
                    
                    elif suggestion['type'] == 'outliers':
                        col_name = suggestion['columns'][0]
                        
                        fig_box = viz_agent.create_visualization(
                            data, 'box',
                            y=col_name,
                            title=f'Outlier Analysis: {col_name}'
                        )
                        st.plotly_chart(fig_box, use_container_width=True)
                        
                        # Show outlier details
                        with st.expander("🎯 Outlier Details"):
                            Q1 = data[col_name].quantile(0.25)
                            Q3 = data[col_name].quantile(0.75)
                            IQR = Q3 - Q1
                            outliers = data[
                                (data[col_name] < (Q1 - 1.5 * IQR)) | 
                                (data[col_name] > (Q3 + 1.5 * IQR))
                            ]
                            
                            st.write(f"Found **{len(outliers)}** outliers")
                            st.dataframe(outliers[[col_name]].head(10), use_container_width=True)
                    
                    # Action buttons
                    col1, col2 = st.columns([3, 1])
                    with col2:
                        if st.button(f"💾 Save Insight #{idx}", key=f"save_{idx}"):
                            st.success(f"Insight #{idx} saved!")
                
            except Exception as e:
                st.error(f"Error performing smart analysis: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
       

# Run the application
if __name__ == "__main__":
    main()
