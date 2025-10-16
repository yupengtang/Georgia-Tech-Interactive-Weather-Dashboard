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
    page_title="TFT Weather Dashboard - Specific Dataset",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_specific_data():
    """Load the specific time series data with caching."""
    grid_file = "TFT_timeseries_predictions_20251016_055154.csv"
    station_file = "TFT_timeseries_predictions_20251016_055154_stations.csv"
    
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

def create_station_comparison(station_df, station_id=None):
    """Create station comparison plots."""
    
    if station_df is None:
        st.warning("No station data available")
        return None
    
    # Filter by station if specified
    if station_id is not None:
        station_data = station_df[station_df['station_id'] == station_id].copy()
        title_suffix = f" - Station {station_id}"
    else:
        station_data = station_df.copy()
        title_suffix = " - All Stations"
    
    if len(station_data) == 0:
        st.warning(f"No data found for station {station_id}")
        return None
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Temperature: Predicted vs Actual', 'Humidity: Predicted vs Actual',
                       'Temperature Time Series', 'Humidity Time Series'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Temperature predictions vs actual
    if 'station_actual_temp' in station_data.columns:
        valid_temp = station_data.dropna(subset=['station_pred_temp', 'station_actual_temp'])
        if len(valid_temp) > 0:
            fig.add_trace(
                go.Scatter(
                    x=valid_temp['station_actual_temp'],
                    y=valid_temp['station_pred_temp'],
                    mode='markers',
                    name='Temperature',
                    marker=dict(color='red', size=6, opacity=0.6)
                ),
                row=1, col=1
            )
            
            # Add perfect prediction line
            min_val = valid_temp['station_actual_temp'].min()
            max_val = valid_temp['station_actual_temp'].max()
            fig.add_trace(
                go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    name='Perfect Prediction',
                    line=dict(color='black', dash='dash')
                ),
                row=1, col=1
            )
    
    # Humidity predictions vs actual
    if 'station_actual_rh' in station_data.columns:
        valid_rh = station_data.dropna(subset=['station_pred_rh', 'station_actual_rh'])
        if len(valid_rh) > 0:
            fig.add_trace(
                go.Scatter(
                    x=valid_rh['station_actual_rh'],
                    y=valid_rh['station_pred_rh'],
                    mode='markers',
                    name='Humidity',
                    marker=dict(color='blue', size=6, opacity=0.6)
                ),
                row=1, col=2
            )
            
            # Add perfect prediction line
            min_val = valid_rh['station_actual_rh'].min()
            max_val = valid_rh['station_actual_rh'].max()
            fig.add_trace(
                go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    name='Perfect Prediction',
                    line=dict(color='black', dash='dash')
                ),
                row=1, col=2
            )
    
    # Time series
    temp_series = station_data.groupby('datetime')['station_pred_temp'].mean().reset_index()
    rh_series = station_data.groupby('datetime')['station_pred_rh'].mean().reset_index()
    
    fig.add_trace(
        go.Scatter(
            x=temp_series['datetime'],
            y=temp_series['station_pred_temp'],
            mode='lines',
            name='Temperature TS',
            line=dict(color='red', width=2)
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=rh_series['datetime'],
            y=rh_series['station_pred_rh'],
            mode='lines',
            name='Humidity TS',
            line=dict(color='blue', width=2)
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        height=800,
        title=f'Station Comparison{title_suffix}',
        showlegend=False
    )
    
    return fig

def create_heatmap_plot(grid_df, variable):
    """Create a heatmap-style plot showing spatial distribution."""
    
    # Get unique time points
    time_points = sorted(grid_df['datetime'].unique())
    
    # Create a grid for heatmap
    lon_min, lon_max = grid_df['longitude'].min(), grid_df['longitude'].max()
    lat_min, lat_max = grid_df['latitude'].min(), grid_df['latitude'].max()
    
    # Create regular grid
    lon_grid = np.linspace(lon_min, lon_max, 50)
    lat_grid = np.linspace(lat_min, lat_max, 50)
    
    # For each time point, create a heatmap
    fig = go.Figure()
    
    # Add frames for animation
    frames = []
    for i, time_point in enumerate(time_points):
        time_data = grid_df[grid_df['datetime'] == time_point]
        
        # Create heatmap data
        heatmap_data = np.zeros((len(lat_grid), len(lon_grid)))
        
        for _, row in time_data.iterrows():
            # Find closest grid point
            lon_idx = np.argmin(np.abs(lon_grid - row['longitude']))
            lat_idx = np.argmin(np.abs(lat_grid - row['latitude']))
            heatmap_data[lat_idx, lon_idx] = row[variable]
        
        frame = go.Frame(
            data=[go.Heatmap(
                z=heatmap_data,
                x=lon_grid,
                y=lat_grid,
                colorscale='Viridis' if 'Temp' in variable else 'Plasma',
                showscale=True
            )],
            name=str(time_point)
        )
        frames.append(frame)
    
    # Add initial frame
    if len(time_points) > 0:
        initial_data = grid_df[grid_df['datetime'] == time_points[0]]
        heatmap_data = np.zeros((len(lat_grid), len(lon_grid)))
        
        for _, row in initial_data.iterrows():
            lon_idx = np.argmin(np.abs(lon_grid - row['longitude']))
            lat_idx = np.argmin(np.abs(lat_grid - row['latitude']))
            heatmap_data[lat_idx, lon_idx] = row[variable]
        
        fig.add_trace(go.Heatmap(
            z=heatmap_data,
            x=lon_grid,
            y=lat_grid,
            colorscale='Viridis' if 'Temp' in variable else 'Plasma',
            showscale=True
        ))
    
    fig.frames = frames
    
    # Add animation controls
    fig.update_layout(
        title=f'{variable.replace("KrigingPrediction_", "")} Heatmap Animation',
        xaxis_title="Longitude",
        yaxis_title="Latitude",
        height=600,
        updatemenus=[{
            "buttons": [
                {
                    "args": [None, {"frame": {"duration": 500, "redraw": True},
                                   "fromcurrent": True, "transition": {"duration": 300}}],
                    "label": "Play",
                    "method": "animate"
                },
                {
                    "args": [[None], {"frame": {"duration": 0, "redraw": True},
                                     "mode": "immediate", "transition": {"duration": 0}}],
                    "label": "Pause",
                    "method": "animate"
                }
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 87},
            "showactive": False,
            "type": "buttons",
            "x": 0.1,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top"
        }]
    )
    
    return fig

def main():
    """Main Streamlit app."""
    
    # Title
    st.title("🌡️ TFT Weather Dashboard - Specific Dataset")
    st.markdown("Interactive visualization for TFT_timeseries_predictions_20251016_055154.csv")
    
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
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🗺️ Spatial Map", "📈 Time Series", "🏢 Station Analysis", "🔥 Heatmap Animation", "📊 Statistics"])
    
    with tab1:
        st.header("Spatial Map Visualization")
        
        # Time selection
        time_points = sorted(grid_df['datetime'].unique())
        selected_time = st.selectbox(
            "Select Time Point:",
            time_points,
            index=len(time_points)-1,  # Default to last time point
            format_func=lambda x: x.strftime("%Y-%m-%d %H:%M")
        )
        
        # Variable selection
        variable = st.selectbox(
            "Select Variable:",
            ['KrigingPrediction_Temp', 'KrigingPrediction_RH'],
            format_func=lambda x: x.replace('KrigingPrediction_', '')
        )
        
        # Show stations option
        show_stations = st.checkbox("Show Weather Stations", value=True)
        
        # Create and display map
        if st.button("Generate Map", type="primary"):
            with st.spinner("Generating map..."):
                fig = create_spatial_map(grid_df, selected_time, variable, show_stations, station_df)
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
        st.header("Station Analysis")
        
        if station_df is not None:
            # Station selection
            stations = sorted(station_df['station_id'].unique())
            selected_station = st.selectbox(
                "Select Station:",
                ['All Stations'] + stations,
                index=0
            )
            
            station_id = None if selected_station == 'All Stations' else selected_station
            
            if st.button("Generate Station Analysis", type="primary"):
                with st.spinner("Generating station analysis..."):
                    fig = create_station_comparison(station_df, station_id)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No station data available")
    
    with tab4:
        st.header("Heatmap Animation")
        
        variable = st.selectbox(
            "Select Variable for Animation:",
            ['KrigingPrediction_Temp', 'KrigingPrediction_RH'],
            format_func=lambda x: x.replace('KrigingPrediction_', ''),
            key="heatmap_variable"
        )
        
        if st.button("Generate Heatmap Animation", type="primary"):
            with st.spinner("Generating heatmap animation..."):
                fig = create_heatmap_plot(grid_df, variable)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
    
    with tab5:
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
