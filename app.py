# ============================================================================
# Parallel Text Processing System - Student Project
# Team Members: Anurag and Abhijith
# 
# This application demonstrates parallel computing concepts including:
# - MapReduce for batch processing
# - Stream processing with sliding windows  
# - Performance benchmarking and analysis
# ============================================================================

import streamlit as st
import pandas as pd
import time
import json
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from multiprocessing import cpu_count
import os

# Import custom modules developed by the team
from processors.mapreduce import MapReduceProcessor           # Implemented by: Anurag
from processors.stream_processor import StreamProcessor       # Implemented by: Abhijith
from processors.sentiment_analyzer import SentimentAnalyzer  # Implemented by: Anurag
from utils.data_loader import DataLoader                     # Implemented by: Abhijith
from utils.performance_monitor import PerformanceMonitor    # Implemented by: Anurag
from utils.visualizer import Visualizer                     # Implemented by: Abhijith

# Page configuration - Basic Streamlit setup
st.set_page_config(
    page_title="Parallel Text Processing System",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'processing_results' not in st.session_state:
    st.session_state.processing_results = {}
if 'performance_data' not in st.session_state:
    st.session_state.performance_data = []
if 'stream_data' not in st.session_state:
    st.session_state.stream_data = []

def main():
    # Main application function - Developed by: Anurag and Abhijith
    st.title("Parallel Text Processing System")
    st.markdown("### Demonstrating MapReduce, Stream Processing, and Performance Benchmarking")
    st.markdown("**Team Members:** Anurag and Abhijith")
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Processing parameters
    num_workers = st.sidebar.slider("Number of Workers", 1, cpu_count(), cpu_count()//2)
    batch_size = st.sidebar.slider("Batch Size", 100, 10000, 1000)
    chunk_size = st.sidebar.slider("Chunk Size", 10, 1000, 100)
    
    # Create tabs for different functionalities
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Data Ingestion", 
        "MapReduce Processing", 
        "Stream Processing", 
        "Performance Analysis", 
        "Real-time Dashboard"
    ])
    
    with tab1:
        data_ingestion_tab(num_workers, batch_size, chunk_size)
    
    with tab2:
        mapreduce_tab(num_workers, batch_size, chunk_size)
    
    with tab3:
        stream_processing_tab(num_workers)
    
    with tab4:
        performance_analysis_tab()
    
    with tab5:
        dashboard_tab()

def data_ingestion_tab(num_workers, batch_size, chunk_size):
    # Data Ingestion Tab - Developed by: Abhijith
    st.header("Data Ingestion")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Data Sources")
        
        # File upload
        uploaded_files = st.file_uploader(
            "Upload text files", 
            type=['txt', 'csv', 'json'], 
            accept_multiple_files=True
        )
        
        # Sample datasets
        st.subheader("Sample Datasets")
        use_sample = st.checkbox("Use sample datasets")
        sample_options = []
    
    with col2:
        st.subheader("Data Summary")
        
        if uploaded_files or use_sample:
            data_loader = DataLoader()
            all_data = []
            
            # Load uploaded files
            if uploaded_files:
                for file in uploaded_files:
                    data = data_loader.load_file(file)
                    if data:
                        all_data.extend(data)
            
            # Load sample datasets if selected
            if use_sample:
                sample_options = st.multiselect(
                    "Select sample datasets",
                    ["Sample Text File", "Sample Tweets CSV"],
                    default=["Sample Text File"]
                )
                
                for option in sample_options:
                    if option == "Sample Text File":
                        data = data_loader.load_sample_text()
                        all_data.extend(data)
                    elif option == "Sample Tweets CSV":
                        data = data_loader.load_sample_tweets()
                        all_data.extend(data)
            
            if all_data:
                st.metric("Total Documents", len(all_data))
                st.metric("Total Words", sum(len(text.split()) for text in all_data))
                st.metric("Average Words per Document", 
                         round(sum(len(text.split()) for text in all_data) / len(all_data), 2))
                
                # Store in session state
                st.session_state.current_data = all_data
                st.success(f"Loaded {len(all_data)} documents successfully!")
            else:
                st.warning("No data loaded. Please upload files or select sample datasets.")
        else:
            st.info("Upload files or select sample datasets to see summary.")

def mapreduce_tab(num_workers, batch_size, chunk_size):
    # MapReduce Processing Tab - Developed by: Anurag
    st.header("MapReduce Processing")
    
    if 'current_data' not in st.session_state:
        st.warning("Please load data in the Data Ingestion tab first.")
        return
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Processing Options")
        
        processing_type = st.selectbox(
            "Select processing type",
            ["Word Count", "Keyword Analysis", "Sentiment Analysis"]
        )
        
        if processing_type == "Keyword Analysis":
            keywords = st.text_input("Enter keywords (comma-separated)", "python,data,analysis")
            keyword_list = [k.strip().lower() for k in keywords.split(",") if k.strip()]
        else:
            keyword_list = []
        
        compare_sequential = st.checkbox("Compare with sequential processing", True)
        
        if st.button("Start MapReduce Processing", type="primary"):
            process_mapreduce(processing_type, num_workers, batch_size, chunk_size, 
                            keyword_list, compare_sequential)
    
    with col2:
        st.subheader("Processing Results")
        
        if 'mapreduce_results' in st.session_state.processing_results:
            results = st.session_state.processing_results['mapreduce_results']
            
            # Display performance metrics
            st.subheader("Performance Metrics")
            
            if 'parallel_time' in results and 'sequential_time' in results:
                col_p, col_s, col_sp = st.columns(3)
                with col_p:
                    st.metric("Parallel Time", f"{results['parallel_time']:.2f}s")
                with col_s:
                    st.metric("Sequential Time", f"{results['sequential_time']:.2f}s")
                with col_sp:
                    speedup = results['sequential_time'] / results['parallel_time']
                    st.metric("Speedup", f"{speedup:.2f}x")
            
            # Display results based on processing type
            if processing_type == "Word Count" and 'word_counts' in results:
                st.subheader("Top Words")
                # Create DataFrame with proper column structure
                word_items = list(results['word_counts'].items())
                if word_items:
                    word_df = pd.DataFrame(word_items)
                    word_df.columns = ['Word', 'Count']
                else:
                    word_df = pd.DataFrame({'Word': [], 'Count': []})
                if not word_df.empty:
                    word_df = word_df.sort_values('Count', ascending=False).head(20)
                
                st.dataframe(word_df, use_container_width=True)
                
                # Visualization
                if not word_df.empty:
                    fig = px.bar(word_df.head(10), x='Word', y='Count', 
                               title="Top 10 Most Frequent Words")
                    st.plotly_chart(fig, use_container_width=True)
            
            elif processing_type == "Sentiment Analysis" and 'sentiment_results' in results:
                st.subheader("Sentiment Analysis Results")
                sentiment_df = pd.DataFrame(results['sentiment_results'])
                
                # Sentiment distribution
                sentiment_counts = sentiment_df['sentiment'].value_counts()
                fig = px.pie(values=sentiment_counts.values, names=sentiment_counts.index,
                           title="Sentiment Distribution")
                st.plotly_chart(fig, use_container_width=True)
                
                # Average polarity
                avg_polarity = sentiment_df['polarity'].mean()
                st.metric("Average Polarity", f"{avg_polarity:.3f}")

def stream_processing_tab(num_workers):
    # Stream Processing Tab - Developed by: Abhijith
    st.header("Stream Processing Simulation")
    
    if 'current_data' not in st.session_state:
        st.warning("Please load data in the Data Ingestion tab first.")
        return
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Stream Configuration")
        
        window_size = st.slider("Window Size (seconds)", 5, 60, 10)
        processing_rate = st.slider("Processing Rate (docs/sec)", 1, 100, 10)
        
        stream_operation = st.selectbox(
            "Stream Operation",
            ["Word Frequency", "Trending Keywords", "Sentiment Monitoring"]
        )
        
        if st.button("Start Stream Processing", type="primary"):
            start_stream_processing(window_size, processing_rate, stream_operation, num_workers)
    
    with col2:
        st.subheader("Real-time Stream Results")
        
        # Placeholder for real-time updates
        stream_placeholder = st.empty()
        
        if st.session_state.stream_data:
            # Display latest window results
            latest_data = st.session_state.stream_data[-1] if st.session_state.stream_data else {}
            
            if stream_operation == "Word Frequency" and 'word_freq' in latest_data:
                # Create DataFrame for word frequency
                word_items = list(latest_data['word_freq'].items())
                if word_items:
                    word_df = pd.DataFrame(word_items)
                    word_df.columns = ['Word', 'Frequency']
                else:
                    word_df = pd.DataFrame({'Word': [], 'Frequency': []})
                if not word_df.empty:
                    word_df = word_df.sort_values('Frequency', ascending=False).head(10)
                
                if not word_df.empty:
                    fig = px.bar(word_df, x='Word', y='Frequency',
                               title=f"Top Words in Last {window_size}s Window")
                    stream_placeholder.plotly_chart(fig, use_container_width=True)
            
            elif stream_operation == "Sentiment Monitoring" and 'sentiment_trend' in latest_data:
                sentiment_df = pd.DataFrame(st.session_state.stream_data)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=sentiment_df['timestamp'],
                    y=sentiment_df['avg_sentiment'],
                    mode='lines+markers',
                    name='Average Sentiment'
                ))
                fig.update_layout(title="Sentiment Trend Over Time")
                stream_placeholder.plotly_chart(fig, use_container_width=True)

def performance_analysis_tab():
    # Performance Analysis Tab - Developed by: Anurag
    st.header("Performance Analysis")
    
    if not st.session_state.performance_data:
        st.info("Run processing tasks to see performance analysis.")
        return
    
    # Performance visualization
    visualizer = Visualizer()
    
    # Throughput analysis
    st.subheader("Throughput Analysis")
    throughput_fig = visualizer.create_throughput_chart(st.session_state.performance_data)
    st.plotly_chart(throughput_fig, use_container_width=True)
    
    # Latency analysis
    st.subheader("Latency Analysis")
    latency_fig = visualizer.create_latency_chart(st.session_state.performance_data)
    st.plotly_chart(latency_fig, use_container_width=True)
    
    # Scalability analysis
    st.subheader("Scalability Analysis")
    scalability_fig = visualizer.create_scalability_chart(st.session_state.performance_data)
    st.plotly_chart(scalability_fig, use_container_width=True)
    
    # Performance summary table
    st.subheader("Performance Summary")
    perf_df = pd.DataFrame(st.session_state.performance_data)
    st.dataframe(perf_df, use_container_width=True)

def dashboard_tab():
    # Real-time Dashboard Tab - Developed by: Abhijith
    st.header("Real-time Dashboard")
    
    # Auto-refresh option
    auto_refresh = st.checkbox("Auto-refresh (5s intervals)")
    
    if auto_refresh:
        time.sleep(5)
        st.rerun()
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if 'current_data' in st.session_state:
            st.metric("Documents Loaded", len(st.session_state.current_data))
        else:
            st.metric("Documents Loaded", 0)
    
    with col2:
        if st.session_state.performance_data:
            avg_throughput = sum(p.get('throughput', 0) for p in st.session_state.performance_data) / len(st.session_state.performance_data)
            st.metric("Avg Throughput", f"{avg_throughput:.2f} docs/s")
        else:
            st.metric("Avg Throughput", "0 docs/s")
    
    with col3:
        if st.session_state.performance_data:
            avg_latency = sum(p.get('latency', 0) for p in st.session_state.performance_data) / len(st.session_state.performance_data)
            st.metric("Avg Latency", f"{avg_latency:.3f}s")
        else:
            st.metric("Avg Latency", "0s")
    
    with col4:
        st.metric("CPU Cores", cpu_count())
    
    # Recent activity
    st.subheader("Recent Processing Activity")
    if st.session_state.performance_data:
        recent_data = st.session_state.performance_data[-10:]  # Last 10 activities
        activity_df = pd.DataFrame(recent_data)
        st.dataframe(activity_df, use_container_width=True)
    else:
        st.info("No recent processing activity.")

def process_mapreduce(processing_type, num_workers, batch_size, chunk_size, keyword_list, compare_sequential):
    """Process data using MapReduce with performance monitoring"""
    data = st.session_state.current_data
    
    with st.spinner("Processing with MapReduce..."):
        processor = MapReduceProcessor(num_workers)
        monitor = PerformanceMonitor()
        
        results = {}
        parallel_time = 0  # Initialize parallel_time
        
        if processing_type == "Word Count":
            # Parallel processing
            start_time = time.time()
            word_counts = processor.word_count_parallel(data, chunk_size)
            parallel_time = time.time() - start_time
            
            results['word_counts'] = dict(word_counts.most_common(100))
            results['parallel_time'] = parallel_time
            
            # Sequential comparison
            if compare_sequential:
                start_time = time.time()
                seq_word_counts = processor.word_count_sequential(data)
                sequential_time = time.time() - start_time
                results['sequential_time'] = sequential_time
        
        elif processing_type == "Sentiment Analysis":
            analyzer = SentimentAnalyzer()
            
            # Parallel processing
            start_time = time.time()
            sentiment_results = analyzer.analyze_parallel(data, num_workers)
            parallel_time = time.time() - start_time
            
            results['sentiment_results'] = sentiment_results
            results['parallel_time'] = parallel_time
            
            # Sequential comparison
            if compare_sequential:
                start_time = time.time()
                seq_sentiment_results = analyzer.analyze_sequential(data)
                sequential_time = time.time() - start_time
                results['sequential_time'] = sequential_time
        
        elif processing_type == "Keyword Analysis":
            # Parallel processing
            start_time = time.time()
            keyword_counts = processor.keyword_analysis_parallel(data, keyword_list, chunk_size)
            parallel_time = time.time() - start_time
            
            results['keyword_counts'] = keyword_counts
            results['parallel_time'] = parallel_time
        
        # Store results and performance data
        st.session_state.processing_results['mapreduce_results'] = results
        
        # Record performance metrics only if we have valid parallel_time
        if parallel_time > 0 and len(data) > 0:
            throughput = len(data) / parallel_time
            performance_record = {
                'timestamp': datetime.now(),
                'operation': f"MapReduce {processing_type}",
                'num_workers': num_workers,
                'documents': len(data),
                'processing_time': parallel_time,
                'throughput': throughput,
                'latency': parallel_time / len(data)
            }
            
            if compare_sequential and 'sequential_time' in results:
                performance_record['sequential_time'] = results['sequential_time']
                performance_record['speedup'] = results['sequential_time'] / parallel_time if parallel_time > 0 else 0
            
            st.session_state.performance_data.append(performance_record)

def start_stream_processing(window_size, processing_rate, stream_operation, num_workers):
    """Start stream processing simulation"""
    data = st.session_state.current_data
    
    stream_processor = StreamProcessor(window_size, processing_rate)
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Simulate streaming for a shorter time to prevent hanging
    simulation_time = 10  # Reduced from 30 to 10 seconds
    
    try:
        status_text.text("Starting stream processing simulation...")
        progress_bar.progress(10)
        
        results = stream_processor.simulate_stream(data, simulation_time, stream_operation, num_workers)
        progress_bar.progress(70)
        
        # Store results
        st.session_state.stream_data.extend(results)
        progress_bar.progress(90)
        
        # Record performance
        results_count = len(results)
        if results_count > 0:
            for i, result in enumerate(results[:10]):  # Limit to first 10 results
                performance_record = {
                    'timestamp': result.get('timestamp', datetime.now()),
                    'operation': f"Stream {stream_operation}",
                    'num_workers': num_workers,
                    'window_size': window_size,
                    'processing_rate': processing_rate,
                    'documents_processed': result.get('documents_processed', 1),
                    'processing_time': result.get('processing_time', 0.1),
                    'throughput': processing_rate,
                    'latency': result.get('processing_time', 0)
                }
                st.session_state.performance_data.append(performance_record)
        
        progress_bar.progress(100)
        status_text.text(f"Stream processing completed! Processed {results_count} data points.")
        
    except Exception as e:
        status_text.text(f"Stream processing error: {str(e)}")
        st.error("Stream processing encountered an issue. Please try again with different parameters.")
        
    finally:
        # Clean up progress indicators
        import time
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()

if __name__ == "__main__":
    main()
