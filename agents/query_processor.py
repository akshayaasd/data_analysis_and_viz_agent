import json
from typing import Dict, Any, List, Optional
import streamlit as st
from config import Config

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

class QueryProcessor:
    def __init__(self):
        self.client = None
        if GROQ_AVAILABLE and Config.GROQ_API_KEY:
            try:
                # Updated Groq client initialization
                self.client = Groq(
                    api_key=Config.GROQ_API_KEY
                )
                st.success("✅ Groq API Connected")
            except Exception as e:
                st.error(f"Failed to initialize Groq client: {str(e)}")
                st.warning("Falling back to basic query processing")
                self.client = None
        elif not Config.GROQ_API_KEY:
            st.warning("⚠️ Groq API key not found. Some AI features will be limited.")
        else:
            st.warning("⚠️ Groq library not available. Install with: pip install groq")
    
    def process_query(self, query: str, columns: List[str], data_summary: Dict) -> Dict[str, Any]:
        """Process natural language query and extract intent and parameters"""
        if not self.client:
            return self._fallback_query_processing(query, columns)
        
        try:
            prompt = self._create_query_analysis_prompt(query, columns, data_summary)
            
            # Updated API call for newer Groq versions
            completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model=Config.GROQ_MODEL,
                temperature=0.1,
                max_tokens=1000,
                top_p=1,
                stream=False,
                stop=None
            )
            
            result = json.loads(completion.choices[0].message.content)
            return result
            
        except json.JSONDecodeError as e:
            st.error(f"Error parsing AI response: {str(e)}")
            return self._fallback_query_processing(query, columns)
        except Exception as e:
            st.error(f"Error processing query with Groq: {str(e)}")
            return self._fallback_query_processing(query, columns)
    
    def _create_query_analysis_prompt(self, query: str, columns: List[str], data_summary: Dict) -> str:
        """Create prompt for query analysis"""
        return f"""
You are a data analysis query processor. Analyze the following query and return ONLY a valid JSON response with the analysis intent and parameters.

Query: "{query}"
Available Columns: {columns}
Data Summary: {data_summary}

Return ONLY a JSON object with this exact structure (no additional text):
{{
    "intent": "descriptive",
    "analysis_type": "summary",
    "target_columns": ["column_name"],
    "groupby_columns": null,
    "aggregation": "mean",
    "visualization_type": "bar",
    "filters": null,
    "time_column": null,
    "explanation": "Brief explanation of what analysis to perform"
}}

Rules:
- intent must be one of: "descriptive", "comparative", "predictive", "visualization", "correlation"
- analysis_type must be one of: "summary", "groupby", "trend", "forecast", "relationship"
- target_columns must be valid column names from the available columns
- aggregation must be one of: "mean", "sum", "count", "min", "max" or null
- visualization_type must be one of: "bar", "line", "scatter", "histogram", "box", "heatmap", "pie" or null
- Only use column names that exist in the available columns list

Examples:
Query: "What is the average sales by region?"
{{
    "intent": "descriptive",
    "analysis_type": "groupby", 
    "target_columns": ["sales"],
    "groupby_columns": ["region"],
    "aggregation": "mean",
    "visualization_type": "bar",
    "filters": null,
    "time_column": null,
    "explanation": "Calculate average sales grouped by region"
}}

Query: "Show me sales trend over time"
{{
    "intent": "visualization",
    "analysis_type": "trend",
    "target_columns": ["sales"],
    "groupby_columns": null,
    "aggregation": null,
    "visualization_type": "line",
    "filters": null,
    "time_column": "date",
    "explanation": "Display sales trend over time as line chart"
}}
        """
    
    def _fallback_query_processing(self, query: str, columns: List[str]) -> Dict[str, Any]:
        """Fallback query processing without AI"""
        query_lower = query.lower()
        
        # Simple keyword-based intent detection
        if any(word in query_lower for word in ['average', 'mean', 'sum', 'total', 'count']):
            intent = "descriptive"
            analysis_type = "summary"
            aggregation = "mean" if 'average' in query_lower or 'mean' in query_lower else "sum" if 'sum' in query_lower or 'total' in query_lower else "count"
        elif any(word in query_lower for word in ['compare', 'vs', 'versus', 'by', 'group']):
            intent = "comparative" 
            analysis_type = "groupby"
            aggregation = "mean"
        elif any(word in query_lower for word in ['trend', 'over time', 'timeline']):
            intent = "visualization"
            analysis_type = "trend"
            aggregation = None
        elif any(word in query_lower for word in ['show', 'plot', 'chart', 'graph']):
            intent = "visualization"
            analysis_type = "summary"
            aggregation = None
        elif any(word in query_lower for word in ['correlation', 'correlate', 'relationship']):
            intent = "correlation"
            analysis_type = "relationship"
            aggregation = None
        else:
            intent = "descriptive"
            analysis_type = "summary"
            aggregation = "mean"
        
        # Try to identify target columns from query
        target_columns = []
        for col in columns:
            if col.lower() in query_lower or col.replace('_', ' ').lower() in query_lower:
                target_columns.append(col)
        
        # If no specific columns found, use first numeric column if available
        if not target_columns:
            numeric_cols = [col for col in columns if any(keyword in col.lower() for keyword in ['sales', 'price', 'amount', 'value', 'count', 'revenue', 'profit'])]
            if numeric_cols:
                target_columns = [numeric_cols[0]]
        
        # Try to identify groupby columns
        groupby_columns = None
        if analysis_type == "groupby":
            for col in columns:
                if any(keyword in col.lower() for keyword in ['region', 'category', 'type', 'group', 'class', 'department']) and col not in target_columns:
                    groupby_columns = [col]
                    break
        
        return {
            "intent": intent,
            "analysis_type": analysis_type,
            "target_columns": target_columns,
            "groupby_columns": groupby_columns,
            "aggregation": aggregation,
            "visualization_type": "bar" if intent == "visualization" else None,
            "filters": None,
            "time_column": None,
            "explanation": f"Fallback processing for: {query}"
        }
    
    def generate_insights(self, analysis_results: Dict, query: str) -> str:
        """Generate natural language insights from analysis results"""
        if not self.client:
            return self._fallback_insights_generation(analysis_results)
        
        try:
            prompt = f"""
Generate clear, actionable insights based on the following analysis results.

Original Query: {query}
Analysis Results: {json.dumps(analysis_results, default=str, indent=2)}

Provide:
1. Key findings (2-3 bullet points)
2. Notable patterns or trends
3. Actionable recommendations if applicable

Keep the response concise, professional, and focused on business value.
            """
            
            completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                model=Config.GROQ_MODEL,
                temperature=0.3,
                max_tokens=500,
                top_p=1,
                stream=False,
                stop=None
            )
            
            return completion.choices[0].message.content
            
        except Exception as e:
            return f"Analysis completed successfully. Results shown above. (Note: AI insight generation unavailable: {str(e)})"
    
    def _fallback_insights_generation(self, analysis_results: Dict) -> str:
        """Fallback insights generation without AI"""
        return "Analysis completed successfully. Please review the results and visualizations above for insights."
