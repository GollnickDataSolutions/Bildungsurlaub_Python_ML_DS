import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

def load_data():
    """Load diamond dataset from CSV file."""
    data_path = Path("001_Datasets") / "Diamonds.csv"
    try:
        df = pd.read_csv(data_path)
        return df
    except FileNotFoundError:
        st.error(f"Data file not found at {data_path}")
        return pd.DataFrame()

def main():
    st.set_page_config(
        page_title="Diamond Data Analyzer",
        page_icon="💎",
        layout="wide"
    )
    
    st.title("💎 Diamond Data Analyzer")
    
    # Load data
    df = load_data()
    
    if df.empty:
        st.stop()
    
    # Show raw data info
    st.sidebar.subheader("Dataset Info")
    st.sidebar.write(f"Total records: {len(df):,}")
    st.sidebar.write(f"Columns: {', '.join(df.columns.tolist())}")
    
    # Display raw data
    with st.expander("Show raw data", expanded=False):
        st.dataframe(df, use_container_width=True)

if __name__ == "__main__":
    main()