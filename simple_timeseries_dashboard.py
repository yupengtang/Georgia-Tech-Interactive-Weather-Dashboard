"""
Specific Time Series Weather Dashboard

Interactive dashboard for the specific TFT_timeseries_predictions_simple.csv file.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Georgia Tech Interactive Weather Dashboard",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_specific_data():
    """Load the specific time series data with caching."""
    grid_file = "simple_TFT_kriging_predictions.csv"
    station_file = "TFT_timeseries_predictions_simple_stations.csv"
    
    if not os.path.exists(grid_file):
        st.error(f"Data file not found: {grid_file}")
        return None, None
    
    # Load data
    grid_df = pd.read_csv(grid_file)
    grid_df['datetime'] = pd.to_datetime(grid_df['datetime'])
    
    station_df = None
    if os.path.exists(station_file):
        station_df = pd.read_csv(station_file)
        station_df['datetime'] = pd.to_datetime(station_df['datetime'])
    
    return grid_df, station_df

def create_spatial_map(grid_df, time_point, variable, show_stations=True, station_df=None):
    """Create interactive spatial map using Plotly."""
    
    # Filter data for specific time point
    time_data = grid_df[grid_df['datetime'] == time_point].copy()
    
    if len(time_data) == 0:
        st.warning(f"No data found for time point: {time_point}")
        return None
    
    # Create scatter plot with better visualization
    fig = px.scatter_mapbox(
        time_data,
        lat='latitude',
        lon='longitude',
        color=variable,
        color_continuous_scale='Viridis' if 'Temp' in variable else 'Plasma',
        size_max=20,  # Increase size
        size=[10] * len(time_data),  # Fixed size for all points
        zoom=15,
        mapbox_style="open-street-map",
        title=f'{variable.replace("KrigingPrediction_", "")} - {time_point.strftime("%Y-%m-%d %H:%M")}',
        labels={variable: f'{variable.replace("KrigingPrediction_", "")} Prediction'},
        hover_data={
            'latitude': ':.4f',
            'longitude': ':.4f'
        },
        custom_data=[time_data[variable].round(2)],
        range_color=[time_data[variable].min(), time_data[variable].max()]  # Force full color range
    )
    
    # Update layout for better visibility
    fig.update_traces(
        marker=dict(
            size=12,  # Larger markers
            opacity=0.8  # Slightly transparent
        ),
        hovertemplate=f"<b>{variable.replace('KrigingPrediction_', '')} Prediction</b><br>" +
                     "Latitude: %{lat:.4f}<br>" +
                     "Longitude: %{lon:.4f}<br>" +
                     f"{variable.replace('KrigingPrediction_', '')}: %{{customdata[0]:.2f}}<br>" +
                     "<extra></extra>"
    )
    
    # Add station points if available
    if show_stations and station_df is not None:
        station_time_data = station_df[station_df['datetime'] == time_point]
        if len(station_time_data) > 0:
            fig.add_trace(go.Scattermapbox(
                lat=station_time_data['latitude'],
                lon=station_time_data['longitude'],
                mode='markers',
                marker=dict(
                    size=12,
                    color='red',
                    symbol='triangle-up'
                ),
                name='Weather Stations',
                text=station_time_data['station_id'],
                hovertemplate='<b>Station %{text}</b><br>' +
                            'Lat: %{lat:.4f}<br>' +
                            'Lon: %{lon:.4f}<br>' +
                            '<extra></extra>'
            ))
    
    fig.update_layout(
        height=600,
        margin=dict(l=0, r=0, t=50, b=0)
    )
    
    return fig

def create_time_series_plot(grid_df, longitude, latitude, radius=0.001):
    """Create time series plot for a specific location."""
    
    # Find nearby grid points
    nearby_points = grid_df[
        (abs(grid_df['longitude'] - longitude) <= radius) &
        (abs(grid_df['latitude'] - latitude) <= radius)
    ].copy()
    
    if len(nearby_points) == 0:
        st.warning(f"No data found near ({longitude}, {latitude})")
        return None
    
    # Average nearby points for each time
    time_series = nearby_points.groupby('datetime').agg({
        'KrigingPrediction_Temp': 'mean',
        'KrigingPrediction_RH': 'mean'
    }).reset_index()
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Temperature', 'Relative Humidity'),
        vertical_spacing=0.1
    )
    
    # Add temperature trace
    fig.add_trace(
        go.Scatter(
            x=time_series['datetime'],
            y=time_series['KrigingPrediction_Temp'],
            mode='lines',
            name='Temperature',
            line=dict(color='red', width=2)
        ),
        row=1, col=1
    )
    
    # Add humidity trace
    fig.add_trace(
        go.Scatter(
            x=time_series['datetime'],
            y=time_series['KrigingPrediction_RH'],
            mode='lines',
            name='Humidity',
            line=dict(color='blue', width=2)
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        height=600,
        title=f'Time Series at ({longitude:.4f}, {latitude:.4f})',
        showlegend=False
    )
    
    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_yaxes(title_text="Temperature (°C)", row=1, col=1)
    fig.update_yaxes(title_text="Humidity (%)", row=2, col=1)
    
    return fig


def main():
    """Main Streamlit app."""
    
    # Title
    st.title("Georgia Tech Interactive Weather Dashboard")
    
    # Load data
    with st.spinner("Loading specific dataset..."):
        grid_df, station_df = load_specific_data()
    
    if grid_df is None:
        st.stop()
    
    # Sidebar
    st.sidebar.header("📊 Dataset Info")
    st.sidebar.subheader("📈 Data Overview")
    st.sidebar.metric("Total Records", f"{len(grid_df):,}")
    st.sidebar.metric("Time Points", f"{grid_df['datetime'].nunique():,}")
    st.sidebar.metric("Locations", f"{len(grid_df[['longitude', 'latitude']].drop_duplicates()):,}")
    
    if station_df is not None:
        st.sidebar.metric("Stations", f"{station_df['station_id'].nunique():,}")
    
    # Dataset info
    st.sidebar.subheader("📅 Time Range")
    st.sidebar.write(f"**Start:** {grid_df['datetime'].min().strftime('%Y-%m-%d %H:%M')}")
    st.sidebar.write(f"**End:** {grid_df['datetime'].max().strftime('%Y-%m-%d %H:%M')}")
    st.sidebar.write(f"**Duration:** {grid_df['datetime'].max() - grid_df['datetime'].min()}")
    
    # Explanation for limited time range
    st.sidebar.info("**Note:** This is a demonstration version with limited time range due to computational constraints. Generating predictions for the full dataset would create an extremely large file with millions of data points, requiring significant processing time and storage. This version is for presentation purposes only.")
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["🗺️ Spatial Map", "📈 Time Series", "📊 Statistics"])
    
    with tab1:
        st.header("Spatial Map Visualization")
        
        # Time selection
        time_points = sorted(grid_df['datetime'].unique())
        
        # Find default time (2015-06-01 04:50)
        default_time = pd.to_datetime('2015-06-01 04:50:00')
        default_index = 0
        if default_time in time_points:
            default_index = time_points.index(default_time)
        else:
            # If exact time not found, use the closest one
            time_diffs = [abs((tp - default_time).total_seconds()) for tp in time_points]
            default_index = time_diffs.index(min(time_diffs))
        
        selected_time = st.selectbox(
            "Select Time Point:",
            time_points,
            index=default_index,  # Default to 2015-06-01 04:50
            format_func=lambda x: x.strftime("%Y-%m-%d %H:%M")
        )
        
        # Variable selection
        variable = st.selectbox(
            "Select Variable:",
            ['KrigingPrediction_Temp', 'KrigingPrediction_RH'],
            format_func=lambda x: x.replace('KrigingPrediction_', '')
        )
        
        # Auto-generate map
        with st.spinner("Generating map..."):
            fig = create_spatial_map(grid_df, selected_time, variable, True, station_df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("Time Series Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            longitude = st.number_input(
                "Longitude:",
                value=-84.4,
                min_value=float(grid_df['longitude'].min()),
                max_value=float(grid_df['longitude'].max()),
                step=0.001,
                format="%.4f"
            )
        
        with col2:
            latitude = st.number_input(
                "Latitude:",
                value=33.77,
                min_value=float(grid_df['latitude'].min()),
                max_value=float(grid_df['latitude'].max()),
                step=0.001,
                format="%.4f"
            )
        
        radius = st.slider(
            "Search Radius:",
            min_value=0.0001,
            max_value=0.01,
            value=0.001,
            step=0.0001,
            format="%.4f"
        )
        
        if st.button("Generate Time Series", type="primary"):
            with st.spinner("Generating time series..."):
                fig = create_time_series_plot(grid_df, longitude, latitude, radius)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("Data Statistics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Temperature Statistics")
            temp_stats = grid_df['KrigingPrediction_Temp'].describe()
            st.metric("Mean", f"{temp_stats['mean']:.2f}°C")
            st.metric("Std", f"{temp_stats['std']:.2f}°C")
            st.metric("Min", f"{temp_stats['min']:.2f}°C")
            st.metric("Max", f"{temp_stats['max']:.2f}°C")
        
        with col2:
            st.subheader("Humidity Statistics")
            rh_stats = grid_df['KrigingPrediction_RH'].describe()
            st.metric("Mean", f"{rh_stats['mean']:.2f}%")
            st.metric("Std", f"{rh_stats['std']:.2f}%")
            st.metric("Min", f"{rh_stats['min']:.2f}%")
            st.metric("Max", f"{rh_stats['max']:.2f}%")
        
        # Data distribution
        st.subheader("Data Distribution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_temp = px.histogram(
                grid_df, 
                x='KrigingPrediction_Temp',
                title='Temperature Distribution',
                nbins=50
            )
            st.plotly_chart(fig_temp, use_container_width=True)
        
        with col2:
            fig_rh = px.histogram(
                grid_df, 
                x='KrigingPrediction_RH',
                title='Humidity Distribution',
                nbins=50
            )
            st.plotly_chart(fig_rh, use_container_width=True)

if __name__ == "__main__":
    main()
