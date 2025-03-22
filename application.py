import streamlit as st
import google.generativeai as genai
import pdfplumber
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import re
import os
from PIL import Image
import pytesseract
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
from io import StringIO, BytesIO
import tempfile

# Configure Gemini AI with API key from environment
genai.configure(api_key=os.getenv("GEMINI_API_KEY", "AIzaSyCCZ4KxaJ2JrDv1PL7n1_2rU_iq-4BB_7g"))

# Set up Streamlit page configuration
st.set_page_config(
    page_title="Financial Report Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Helper Functions for Data Extraction and Processing
@st.cache_data
def extract_text_from_pdf(uploaded_file):
    """
    Extracts plain text from PDF pages with improved error handling.
    
    Args:
        uploaded_file: The uploaded PDF file
        
    Returns:
        str: Extracted text or None if extraction failed
    """
    try:
        # Create a temporary file to handle the PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_path = temp_file.name
            
        # Extract text from the PDF
        with pdfplumber.open(temp_path) as pdf:
            all_text = []
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    all_text.append(text)
            
            # Remove temporary file
            os.unlink(temp_path)
            
            if not all_text:
                return None
            return "\n".join(all_text)
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return None

@st.cache_data
def extract_text_from_image(uploaded_file):
    """
    Extracts text from image using OCR with improved error handling.
    
    Args:
        uploaded_file: The uploaded image file
        
    Returns:
        str: Extracted text or None if extraction failed
    """
    try:
        # Open the image using PIL
        image = Image.open(uploaded_file)
        
        # Extract text using pytesseract
        text = pytesseract.image_to_string(image)
        
        if not text.strip():
            st.warning("No text detected in the image.")
            return None
            
        return text
    except Exception as e:
        st.error(f"Error extracting text from image: {str(e)}")
        return None

@st.cache_data
def process_csv_file(uploaded_file):
    """
    Processes a CSV file and returns a DataFrame with enhanced preprocessing.
    
    Args:
        uploaded_file: The uploaded CSV file
        
    Returns:
        tuple: (DataFrame, list of financial columns)
    """
    try:
        # Get the file content
        content = uploaded_file.getvalue().decode("utf-8")
        
        # Try to read CSV with auto-detection of delimiter
        try:
            df = pd.read_csv(StringIO(content))
        except:
            # Try with different delimiters if standard comma fails
            for delimiter in [';', '\t', '|']:
                try:
                    df = pd.read_csv(StringIO(content), delimiter=delimiter)
                    if len(df.columns) > 1:  # Successful parsing
                        break
                except:
                    continue
        
        # Try to identify financial columns automatically
        columns = df.columns.tolist()
        financial_columns = []
        financial_terms = [
            'revenue', 'sales', 'income', 'expense', 'profit', 'loss',
            'assets', 'liabilities', 'equity', 'cash', 'debt', 'ebitda',
            'margin', 'roi', 'eps', 'dividend', 'investment', 'capital'
        ]
        
        for col in columns:
            col_lower = col.lower()
            # Check for financial terms in column name
            if any(term in col_lower for term in financial_terms):
                financial_columns.append(col)
            # Check if column contains mostly numeric values (potential financial data)
            elif df[col].dtype in [np.int64, np.float64] or pd.to_numeric(df[col], errors='coerce').notna().sum() / len(df) >= 0.7:
                financial_columns.append(col)
        
        # If we have date-like columns, add them to the list
        date_columns = []
        for col in columns:
            if df[col].dtype == 'object':
                # Check if column might contain dates
                if any(term in col.lower() for term in ['date', 'year', 'month', 'quarter', 'period']):
                    date_columns.append(col)
                    # Try to convert to datetime if possible
                    try:
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                        if df[col].notna().sum() > 0:  # If some values were successfully converted
                            date_columns.append(col)
                    except:
                        pass
                        
        return df, financial_columns, date_columns
    except Exception as e:
        st.error(f"Error processing CSV file: {str(e)}")
        return None, [], []

def extract_financial_data_from_text(text):
    """
    Extract structured financial data from text with improved pattern matching.
    
    Args:
        text: The text to extract data from
        
    Returns:
        DataFrame: Extracted financial data or None if extraction failed
    """
    try:
        if not text:
            return None
            
        lines = text.split("\n")
        structured_data = []
        
        # Look for common financial statement headers
        financial_sections = {
            'income_statement': ['income statement', 'profit and loss', 'statement of operations', 'p&l'],
            'balance_sheet': ['balance sheet', 'statement of financial position', 'assets and liabilities'],
            'cash_flow': ['cash flow', 'statement of cash flows', 'cash flow statement']
        }
        
        current_section = None
        section_data = []
        
        for i, line in enumerate(lines):
            line_lower = line.lower().strip()
            
            # Detect section headers
            for section, keywords in financial_sections.items():
                if any(keyword in line_lower for keyword in keywords):
                    current_section = section
                    break
            
            # Extract numbers with currency symbols
            currency_values = re.findall(r'[$€£¥][\d,]+\.?\d*|\d+\.?\d*[$€£¥]|[\d,.]+\s*(?:million|billion|m|b)', line_lower)
            
            # Extract years and quarters
            year_match = re.search(r'\b(19|20)\d{2}\b', line)
            quarter_match = re.search(r'Q[1-4]', line)
            
            # Look for financial terms in the line
            financial_terms = ['revenue', 'sales', 'income', 'expense', 'profit', 'loss', 
                            'assets', 'liabilities', 'equity', 'cash', 'debt', 'ebitda']
            
            has_financial_term = any(term in line_lower for term in financial_terms)
            
            # If the line has both financial terms and values, add it to structured data
            if has_financial_term and (currency_values or any(c.isdigit() for c in line)):
                # Extract all numeric values from the line
                values = re.findall(r'[-+]?[0-9]*\.?[0-9]+', line.replace(',', ''))
                numeric_values = [float(val) for val in values if val]
                
                if numeric_values:
                    time_period = ""
                    if year_match:
                        time_period = year_match.group()
                    if quarter_match:
                        time_period += f" {quarter_match.group()}"
                    
                    # Extract the term description
                    term_description = re.sub(r'[\d.,]+', '', line).strip()
                    term_description = re.sub(r'[$€£¥]', '', term_description).strip()
                    
                    structured_data.append({
                        'section': current_section,
                        'description': term_description,
                        'time_period': time_period,
                        'values': numeric_values
                    })
        
        # Convert to more structured DataFrame
        if structured_data:
            # First, convert to a standard format
            rows = []
            for item in structured_data:
                description = item['description']
                time_period = item['time_period']
                
                # Standardize common financial terms
                clean_description = description.lower()
                if any(term in clean_description for term in ['revenue', 'sales']):
                    key = 'Revenue'
                elif any(term in clean_description for term in ['expense', 'cost']):
                    key = 'Expenses'
                elif any(term in clean_description for term in ['profit', 'income', 'earnings']):
                    key = 'Profit'
                elif 'assets' in clean_description:
                    key = 'Assets'
                elif 'liabilities' in clean_description:
                    key = 'Liabilities'
                elif 'equity' in clean_description:
                    key = 'Equity'
                else:
                    key = description  # Keep original if no match
                
                if time_period and len(item['values']) > 0:
                    rows.append({
                        'Time Period': time_period,
                        key: item['values'][0]  # Take first value
                    })
            
            if not rows:
                return None
                
            # Convert to DataFrame and pivot if needed
            df = pd.DataFrame(rows)
            
            # If we have multiple financial metrics, pivot the data
            if len(df.columns) > 2:
                return df
            else:
                # Simple aggregation to handle duplicates
                return df.groupby('Time Period').sum().reset_index()
                
        return None
    except Exception as e:
        st.error(f"Error extracting financial data from text: {str(e)}")
        return None

def preprocess_data(df):
    """
    Cleans and preprocesses the data with improved handling of missing values.
    
    Args:
        df: DataFrame to preprocess
        
    Returns:
        DataFrame: Processed DataFrame or None if processing failed
    """
    try:
        if df is None or df.empty:
            return None
            
        # Make a copy to avoid modifying the original
        processed_df = df.copy()
        
        # Convert column names to string to avoid type errors
        processed_df.columns = processed_df.columns.astype(str)
        
        # Identify numeric columns (potential financial metrics)
        numeric_cols = processed_df.select_dtypes(include=['number']).columns.tolist()
        
        # Drop rows where all financial metrics are missing
        if numeric_cols:
            processed_df = processed_df.dropna(how='all', subset=numeric_cols)
        
        # Fill remaining NaN values with 0 for numeric columns
        for col in numeric_cols:
            processed_df[col] = processed_df[col].fillna(0)
        
        # Remove duplicate rows
        processed_df = processed_df.drop_duplicates()
        
        # Try to extract or create a year column for time series analysis
        if 'Time Period' in processed_df.columns:
            try:
                # Extract year from time period column
                processed_df['Year'] = processed_df['Time Period'].str.extract(r'(\d{4})').astype(float)
                # Sort by year
                processed_df = processed_df.sort_values(by='Year')
            except:
                pass
        
        # If we don't have a Year column yet, try to create one from other date columns
        if 'Year' not in processed_df.columns:
            date_columns = [col for col in processed_df.columns if any(term in col.lower() for term in ['date', 'year', 'period'])]
            for col in date_columns:
                try:
                    # Try to convert to datetime
                    processed_df[col] = pd.to_datetime(processed_df[col], errors='coerce')
                    # Extract year
                    processed_df['Year'] = processed_df[col].dt.year
                    break
                except:
                    continue
        
        return processed_df
    except Exception as e:
        st.error(f"Error preprocessing data: {str(e)}")
        return None

def validate_data(df):
    """
    Validates the extracted data with improved checks.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        bool: True if data is valid, False otherwise
    """
    if df is None:
        st.error("No data found in the file. Please check the file and try again.")
        return False
    
    if df.empty:
        st.error("The extracted data is empty. Please check the file format.")
        return False
    
    # Check if we have enough rows
    if len(df) < 2:
        st.warning("Limited data available (less than 2 rows). Analysis may be limited.")
        return False
    
    # Check if we have numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) < 1:
        st.error("No numeric data found for analysis. Please upload a valid financial report.")
        return False
    
    return True

def generate_ai_insights(df, summarization_length):
    """
    Generate financial insights using AI with optimized prompts.
    
    Args:
        df: DataFrame containing financial data
        summarization_length: "Short" or "Detailed"
        
    Returns:
        str: AI-generated insights
    """
    try:
        # Convert DataFrame to string format optimized for AI processing
        # For better results, include column descriptions
        column_descriptions = [f"- {col}: {df[col].dtype}" for col in df.columns]
        column_desc_text = "\n".join(column_descriptions)
        
        # Include basic statistics for numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        stats_summary = []
        for col in numeric_cols:
            stats = df[col].describe()
            stats_summary.append(f"- {col}: min={stats['min']:.2f}, max={stats['max']:.2f}, mean={stats['mean']:.2f}")
        
        stats_text = "\n".join(stats_summary)
        
        # Include first few rows for context
        data_sample = df.head(10).to_string(index=False)
        
        # Choose prompt template based on summarization length
        if summarization_length == "Short":
            prompt = f"""
            You are a financial analyst. Below is financial data extracted from a report.
            Provide a concise 3-5 bullet point summary of the key insights.
            Focus only on the clearly observable trends and significant findings.
            Be specific with numbers and avoid vague statements.
            
            Column descriptions:
            {column_desc_text}
            
            Statistical summary:
            {stats_text}
            
            Data sample:
            {data_sample}
            
            Your analysis (3-5 bullet points):
            """
        else:  # Detailed
            prompt = f"""
            You are a financial analyst. Below is financial data extracted from a report.
            Provide a detailed analysis covering:
            1. Key financial trends over time (revenue, profit, etc.)
            2. Financial health assessment (assets vs liabilities, equity growth)
            3. Areas of concern or opportunity
            4. Recommendations based on the data
            
            Column descriptions:
            {column_desc_text}
            
            Statistical summary:
            {stats_text}
            
            Data sample:
            {data_sample}
            
            Your detailed analysis:
            """
        
        # Set up the Gemini model for generation
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        # Generate the response with the model
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Error generating AI insights: {str(e)}")
        return "Unable to generate insights due to an error. Please try again."

def generate_future_predictions(df, prediction_period, ml_model):
    """
    Generates predictions for future periods using selected ML model.
    
    Args:
        df: DataFrame containing historical data
        prediction_period: Number of periods to predict
        ml_model: ML model to use ("Linear Regression", "Random Forest", "ARIMA")
        
    Returns:
        DataFrame: Predictions for future periods
    """
    try:
        # Ensure we have the required columns
        required_cols = ['Year']
        if not all(col in df.columns for col in required_cols):
            st.warning("Missing required 'Year' column for prediction.")
            return None
        
        # Find potential target columns (financial metrics)
        numeric_cols = df.select_dtypes(include=['number']).columns
        target_cols = [col for col in numeric_cols if col != 'Year']
        
        if not target_cols:
            st.warning("No financial metrics found for prediction.")
            return None
        
        # Use the first financial metric if 'Revenue' is not available
        target_col = 'Revenue' if 'Revenue' in target_cols else target_cols[0]
        
        # Ensure we have enough data for prediction
        if len(df) < 3:
            st.warning(f"Not enough historical data for reliable {ml_model} prediction. Need at least 3 data points.")
            return None
        
        # Sort data by year to ensure chronological order
        df = df.sort_values(by='Year')
        
        # Prepare data for ML model
        X = df['Year'].values.reshape(-1, 1)
        y = df[target_col].values
        
        # Generate future years for prediction
        last_year = df['Year'].max()
        future_years = np.array([last_year + i for i in range(1, prediction_period + 1)]).reshape(-1, 1)
        
        # Select and train the ML model
        if ml_model == "Linear Regression":
            model = LinearRegression()
            model.fit(X, y)
            predictions = model.predict(future_years)
            
        elif ml_model == "Random Forest":
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)
            predictions = model.predict(future_years)
            
        elif ml_model == "ARIMA":
            # Fit ARIMA model - try a standard (1,1,1) model
            try:
                model = ARIMA(y, order=(1, 1, 1))
                model_fit = model.fit()
                
                # Generate forecast
                forecast = model_fit.forecast(steps=prediction_period)
                predictions = forecast
            except Exception as e:
                st.warning(f"ARIMA model failed: {str(e)}. Using Linear Regression as fallback.")
                # Fallback to Linear Regression
                model = LinearRegression()
                model.fit(X, y)
                predictions = model.predict(future_years)
        
        # Create DataFrame with predictions
        future_df = pd.DataFrame({
            'Year': future_years.flatten(),
            f'Predicted {target_col}': predictions
        })
        
        return future_df
    except Exception as e:
        st.error(f"Error generating predictions: {str(e)}")
        return None

def create_time_series_chart(df, selected_columns):
    """
    Creates an interactive time series chart for selected financial metrics.
    
    Args:
        df: DataFrame containing financial data
        selected_columns: List of columns to include in the chart
        
    Returns:
        plotly Figure: Interactive time series chart
    """
    try:
        # Ensure we have a time column and selected financial metrics
        if 'Year' not in df.columns or not selected_columns:
            st.warning("Missing Year column or financial metrics for visualization.")
            return None
        
        # Create a copy of the dataframe with only relevant columns
        plot_df = df[['Year'] + selected_columns].copy()
        
        # Sort by year
        plot_df = plot_df.sort_values(by='Year')
        
        # Create an empty figure
        fig = go.Figure()
        
        # Add a trace for each selected column
        for column in selected_columns:
            fig.add_trace(
                go.Scatter(
                    x=plot_df['Year'],
                    y=plot_df[column],
                    mode='lines+markers',
                    name=column,
                    line=dict(width=3),
                    marker=dict(size=8)
                )
            )
        
        # Update layout for better visualization
        fig.update_layout(
            title="Financial Metrics Over Time",
            xaxis_title="Year",
            yaxis_title="Value",
            legend_title="Metrics",
            hovermode="x unified",
            template="plotly_white",
            height=500
        )
        
        # Add range slider for interactive time selection
        fig.update_layout(
            xaxis=dict(
                rangeslider=dict(visible=True),
                type="linear"
            )
        )
        
        return fig
    except Exception as e:
        st.error(f"Error creating time series chart: {str(e)}")
        return None

def create_financial_summary_cards(df):
    """
    Creates summary cards for key financial metrics.
    
    Args:
        df: DataFrame containing financial data
        
    Returns:
        None (directly renders cards with st.columns)
    """
    try:
        # Identify potential financial metrics
        key_metrics = ['Revenue', 'Expenses', 'Profit', 'Assets', 'Liabilities', 'Equity']
        available_metrics = [metric for metric in key_metrics if metric in df.columns]
        
        if not available_metrics:
            # If standard metrics not found, use available numeric columns
            available_metrics = df.select_dtypes(include=['number']).columns.tolist()
            available_metrics = [m for m in available_metrics if m != 'Year']
        
        if not available_metrics:
            st.warning("No financial metrics found for summary cards.")
            return
        
        # Calculate growth for each metric
        metrics_with_growth = []
        for metric in available_metrics:
            if len(df) >= 2 and 'Year' in df.columns:
                # Sort by year
                sorted_df = df.sort_values(by='Year')
                latest_value = sorted_df[metric].iloc[-1]
                previous_value = sorted_df[metric].iloc[-2]
                
                if previous_value != 0:
                    growth = (latest_value - previous_value) / previous_value * 100
                else:
                    growth = float('inf') if latest_value > 0 else 0
                
                metrics_with_growth.append((metric, latest_value, growth))
            else:
                # If no year column or not enough data, just use the mean
                metrics_with_growth.append((metric, df[metric].mean(), None))
        
        # Create metrics cards using columns
        cols = st.columns(min(len(metrics_with_growth), 3))
        
        for i, (metric, value, growth) in enumerate(metrics_with_growth):
            col_idx = i % len(cols)
            
            with cols[col_idx]:
                st.metric(
                    label=metric,
                    value=f"${value:,.2f}" if value >= 0 else f"-${abs(value):,.2f}",
                    delta=f"{growth:.1f}%" if growth is not None else None,
                    delta_color="normal" if growth is None else "inverse" if metric in ["Expenses", "Liabilities"] else "normal"
                )
    except Exception as e:
        st.error(f"Error creating summary cards: {str(e)}")

def create_comparison_chart(df, x_axis, y_axis):
    """
    Creates a comparison chart between two financial metrics.
    
    Args:
        df: DataFrame containing financial data
        x_axis: Column to use for x-axis
        y_axis: Column to use for y-axis
        
    Returns:
        plotly Figure: Interactive comparison chart
    """
    try:
        if x_axis not in df.columns or y_axis not in df.columns:
            st.warning(f"Missing columns for comparison chart: {x_axis} or {y_axis}")
            return None
        
        # Create scatter plot
        fig = px.scatter(
            df, 
            x=x_axis, 
            y=y_axis,
            size=abs(df[y_axis]) if len(df) > 3 else None,  # Size points by value magnitude if enough data
            hover_data=df.columns,
            trendline="ols" if len(df) >= 3 else None,  # Add trendline if enough data
            title=f"{y_axis} vs {x_axis}"
        )
        
        # Enhance the layout
        fig.update_layout(
            xaxis_title=x_axis,
            yaxis_title=y_axis,
            template="plotly_white",
            height=500
        )
        
        # Add annotations for highest and lowest points if we have enough data
        if len(df) > 2:
            max_idx = df[y_axis].idxmax()
            min_idx = df[y_axis].idxmin()
            
            # Add annotations
            fig.add_annotation(
                x=df.loc[max_idx, x_axis],
                y=df.loc[max_idx, y_axis],
                text="Highest",
                showarrow=True,
                arrowhead=1
            )
            
            fig.add_annotation(
                x=df.loc[min_idx, x_axis],
                y=df.loc[min_idx, y_axis],
                text="Lowest",
                showarrow=True,
                arrowhead=1
            )
        
        return fig
    except Exception as e:
        st.error(f"Error creating comparison chart: {str(e)}")
        return None

def create_pie_chart(df, metric):
    """
    Creates a pie chart for a financial metric breakdown.
    
    Args:
        df: DataFrame containing financial data
        metric: Column to visualize
        
    Returns:
        plotly Figure: Interactive pie chart
    """
    try:
        if metric not in df.columns:
            st.warning(f"Column {metric} not found for pie chart")
            return None
        
        # Check if we have a 'Year' or 'Time Period' column for labels
        if 'Year' in df.columns:
            labels = df['Year'].astype(str)
        elif 'Time Period' in df.columns:
            labels = df['Time Period']
        else:
            # Create generic labels
            labels = [f"Period {i+1}" for i in range(len(df))]
        
        # Create values for pie chart, using absolute values
        values = df[metric].abs()
        
        # Create pie chart
        fig = px.pie(
            names=labels,
            values=values,
            title=f"{metric} Distribution by Period",
            hole=0.4,  # Create a donut chart
        )
        
        # Improve layout
        fig.update_layout(
            legend_title="Time Period",
            template="plotly_white"
        )
        
        # Add percentage to labels
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label'
        )
        
        return fig
    except Exception as e:
        st.error(f"Error creating pie chart: {str(e)}")
        return None

def create_combined_forecast_chart(historical_df, forecast_dfs, target_col):
    """
    Creates a chart combining historical data with multiple forecasts.
    
    Args:
        historical_df: DataFrame with historical data
        forecast_dfs: Dictionary of forecast DataFrames by model
        target_col: Column to visualize
        
    Returns:
        plotly Figure: Interactive forecast chart
    """
    try:
        if not forecast_dfs:
            st.warning("No forecast data available for visualization")
            return None
        
        # Create a combined figure
        fig = go.Figure()
        
        # First, add historical data
        if 'Year' in historical_df.columns and target_col in historical_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=historical_df['Year'],
                    y=historical_df[target_col],
                    mode='lines+markers',
                    name='Historical Data',
                    line=dict(color='blue', width=3),
                    marker=dict(size=8)
                )
            )
        
        # Add forecast for each model
        colors = ['red', 'green', 'purple', 'orange']
        for i, (model_name, forecast_df) in enumerate(forecast_dfs.items()):
            forecast_col = forecast_df.columns[1]  # Should be the predicted column
            color = colors[i % len(colors)]
            
            fig.add_trace(
                go.Scatter(
                    x=forecast_df['Year'],
                    y=forecast_df[forecast_col],
                    mode='lines+markers',
                    name=f'{model_name} Forecast',
                    line=dict(color=color, width=2, dash='dash'),
                    marker=dict(size=6)
                )
            )
        
        # Update layout for better visualization
        fig.update_layout(
            title=f"Historical Data and Forecasts for {target_col}",
            xaxis_title="Year",
            yaxis_title=target_col,
            legend_title="Data Source",
            hovermode="x unified",
            template="plotly_white",
            height=500
        )
        
        # Add vertical line separating historical and forecast data
        if 'Year' in historical_df.columns:
            last_historical_year = historical_df['Year'].max()
            
            fig.add_shape(
                type="line",
                x0=last_historical_year,
                y0=0,
                x1=last_historical_year,
                y1=1,
                yref="paper",
                line=dict(color="gray", width=2, dash="dot")
            )
            
            fig.add_annotation(
                x=last_historical_year,
                y=1,
                yref="paper",
                text="Forecast Start",
                showarrow=False,
                yshift=10
            )
        
        return fig
    except Exception as e:
        st.error(f"Error creating forecast chart: {str(e)}")
        return None

# Main Application
def main():
    st.title("📊 Financial Report Analyzer")
    st.markdown("""
    Upload financial reports (PDF, CSV, or images) to extract, analyze, and visualize key financial metrics.
    Get AI-powered insights and predictive analytics for your financial data.
    """)
    
    # Initialize session state variables
    if 'current_data' not in st.session_state:
        st.session_state.current_data = None
    
    if 'file_type' not in st.session_state:
        st.session_state.file_type = None
        
    if 'forecast_data' not in st.session_state:
        st.session_state.forecast_data = {}
    
    # Sidebar for file upload and options
    with st.sidebar:
        st.header("Upload & Options")
        
        uploaded_file = st.file_uploader("Upload a financial report", 
                                        type=["pdf", "csv", "jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            file_type = uploaded_file.type
            st.session_state.file_type = file_type
            
            with st.spinner("Processing file..."):
                # Process based on file type
                if file_type == "application/pdf":
                    extracted_text = extract_text_from_pdf(uploaded_file)
                    if extracted_text:
                        # Try to extract structured data
                        df = extract_financial_data_from_text(extracted_text)
                        
                        if df is None:
                            st.error("Could not extract structured financial data from the PDF.")
                            st.session_state.current_data = None
                        else:
                            df = preprocess_data(df)
                            if validate_data(df):
                                st.success("Successfully extracted financial data from PDF!")
                                st.session_state.current_data = df
                            else:
                                st.session_state.current_data = None
                    else:
                        st.error("Could not extract text from the PDF.")
                        st.session_state.current_data = None
                
                elif file_type.startswith("image"):
                    extracted_text = extract_text_from_image(uploaded_file)
                    if extracted_text:
                        # Try to extract structured data
                        df = extract_financial_data_from_text(extracted_text)
                        
                        if df is None:
                            st.error("Could not extract structured financial data from the image.")
                            st.session_state.current_data = None
                        else:
                            df = preprocess_data(df)
                            if validate_data(df):
                                st.success("Successfully extracted financial data from the image!")
                                st.session_state.current_data = df
                            else:
                                st.session_state.current_data = None
                    else:
                        st.error("Could not extract text from the image.")
                        st.session_state.current_data = None
                
                elif file_type == "text/csv":
                    df, financial_columns, date_columns = process_csv_file(uploaded_file)
                    if df is not None:
                        df = preprocess_data(df)
                        if validate_data(df):
                            st.success("Successfully processed CSV data!")
                            st.session_state.current_data = df
                            if financial_columns:
                                st.info(f"Detected financial columns: {', '.join(financial_columns)}")
                        else:
                            st.session_state.current_data = None
                    else:
                        st.error("Could not process the CSV file.")
                        st.session_state.current_data = None
                
                else:
                    st.error("Unsupported file type. Please upload a PDF, CSV, or image file.")
                    st.session_state.current_data = None
        
        # Add options for analysis if data is loaded
        if st.session_state.current_data is not None:
            st.markdown("---")
            st.header("Analysis Options")
            
            # AI Insights options
            st.subheader("AI Insights")
            summarization_length = st.radio(
                "Analysis depth:",
                options=["Short", "Detailed"],
                index=0
            )
            
            # Forecasting options
            st.subheader("Forecasting")
            prediction_period = st.slider("Prediction periods:", 1, 10, 3)
            ml_model = st.selectbox(
                "Forecasting model:",
                options=["Linear Regression", "Random Forest", "ARIMA"],
                index=0
            )
            
            # Generate forecast button
            if st.button("Generate Forecast"):
                with st.spinner("Generating forecast..."):
                    forecast = generate_future_predictions(
                        st.session_state.current_data, 
                        prediction_period, 
                        ml_model
                    )
                    
                    if forecast is not None:
                        st.session_state.forecast_data[ml_model] = forecast
                        st.success(f"Forecast generated with {ml_model}!")
            
            # Download options
            st.markdown("---")
            st.subheader("Download Results")
            
            if st.button("Download Data as CSV"):
                csv = st.session_state.current_data.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="financial_data.csv",
                    mime="text/csv"
                )
    
    # Main content area
    if st.session_state.current_data is not None and not st.session_state.current_data.empty:
        # Create tabs for different visualizations
        tabs = st.tabs(["Data Overview", "Visualizations", "AI Insights", "Forecasting"])
        
        # Data Overview Tab
        with tabs[0]:
            st.header("Financial Data Overview")
            
            # Show summary cards for key metrics
            create_financial_summary_cards(st.session_state.current_data)
            
            # Display the data table
            st.subheader("Data Table")
            st.dataframe(st.session_state.current_data, use_container_width=True)
            
            # Display basic statistics
            st.subheader("Statistical Summary")
            numeric_cols = st.session_state.current_data.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                st.dataframe(st.session_state.current_data[numeric_cols].describe(), use_container_width=True)
            else:
                st.warning("No numeric columns found for statistical summary.")
        
        # Visualizations Tab
        with tabs[1]:
            st.header("Financial Visualizations")
            
            # Identify numeric columns for visualization
            numeric_cols = st.session_state.current_data.select_dtypes(include=['number']).columns.tolist()
            numeric_cols = [col for col in numeric_cols if col != 'Year']  # Exclude Year
            
            if not numeric_cols:
                st.warning("No numeric columns available for visualization.")
            else:
                # Time Series Chart
                st.subheader("Time Series Analysis")
                
                if 'Year' not in st.session_state.current_data.columns:
                    st.warning("No 'Year' column found for time series analysis.")
                else:
                    selected_metrics = st.multiselect(
                        "Select metrics to visualize:",
                        options=numeric_cols,
                        default=numeric_cols[:min(3, len(numeric_cols))]
                    )
                    
                    if selected_metrics:
                        time_series_chart = create_time_series_chart(
                            st.session_state.current_data,
                            selected_metrics
                        )
                        
                        if time_series_chart:
                            st.plotly_chart(time_series_chart, use_container_width=True)
                
                # Comparison Chart
                st.subheader("Metrics Comparison")
                
                if len(numeric_cols) >= 2:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        x_metric = st.selectbox(
                            "Select X-axis metric:",
                            options=numeric_cols,
                            index=0
                        )
                    
                    with col2:
                        y_metric = st.selectbox(
                            "Select Y-axis metric:",
                            options=numeric_cols,
                            index=min(1, len(numeric_cols)-1)
                        )
                    
                    comparison_chart = create_comparison_chart(
                        st.session_state.current_data,
                        x_metric,
                        y_metric
                    )
                    
                    if comparison_chart:
                        st.plotly_chart(comparison_chart, use_container_width=True)
                else:
                    st.warning("Need at least 2 numeric columns for comparison chart.")
                
                # Pie Chart for distribution
                st.subheader("Distribution Analysis")
                
                selected_metric_for_pie = st.selectbox(
                    "Select metric for distribution analysis:",
                    options=numeric_cols,
                    index=0
                )
                
                pie_chart = create_pie_chart(
                    st.session_state.current_data,
                    selected_metric_for_pie
                )
                
                if pie_chart:
                    st.plotly_chart(pie_chart, use_container_width=True)
        
        # AI Insights Tab
        with tabs[2]:
            st.header("AI-Powered Financial Insights")
            
            with st.spinner("Generating AI insights..."):
                insights = generate_ai_insights(
                    st.session_state.current_data,
                    summarization_length
                )
                
                if insights:
                    st.markdown(insights)
                else:
                    st.error("Could not generate AI insights. Please try again.")
        
        # Forecasting Tab
        with tabs[3]:
            st.header("Financial Forecasting")
            
            if not st.session_state.forecast_data:
                st.info("No forecasts generated yet. Use the sidebar to generate forecasts.")
            else:
                # Display each forecast
                for model_name, forecast_df in st.session_state.forecast_data.items():
                    st.subheader(f"{model_name} Forecast")
                    st.dataframe(forecast_df, use_container_width=True)
                
                # Show combined forecast chart
                st.subheader("Combined Forecast Visualization")
                
                # Identify target column for forecasting
                if 'Revenue' in st.session_state.current_data.columns:
                    target_col = 'Revenue'
                else:
                    # Pick the first numeric column that's not Year
                    numeric_cols = st.session_state.current_data.select_dtypes(include=['number']).columns
                    target_col = next((col for col in numeric_cols if col != 'Year'), None)
                
                if target_col:
                    forecast_chart = create_combined_forecast_chart(
                        st.session_state.current_data,
                        st.session_state.forecast_data,
                        target_col
                    )
                    
                    if forecast_chart:
                        st.plotly_chart(forecast_chart, use_container_width=True)
                else:
                    st.warning("Could not identify target column for forecast visualization.")
    else:
        # Show welcome message and instructions if no data is loaded
        st.info("👈 Please upload a financial report file from the sidebar to get started.")
        
        # Show features overview
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📄 Supported File Types")
            st.markdown("""
            - PDF financial reports
            - CSV financial data
            - Images containing financial information
            """)
            
            st.markdown("### 📊 Visualization Features")
            st.markdown("""
            - Interactive time series charts
            - Financial metrics comparison
            - Distribution analysis
            - Custom chart options
            """)
        
        with col2:
            st.markdown("### 🧠 AI-Powered Analysis")
            st.markdown("""
            - Automatic data extraction
            - Intelligent financial insights
            - Key trends identification
            - Actionable recommendations
            """)
            
            st.markdown("### 📈 Forecasting Capabilities")
            st.markdown("""
            - Multiple forecasting models
            - Customizable prediction periods
            - Combined forecast visualization
            - Trend analysis
            """)

if __name__ == "__main__":
    main()
