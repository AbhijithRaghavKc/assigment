# ============================================================================
# Visualizer - Implemented by: Abhijith
# 
# This module creates interactive visualizations for performance metrics and
# analysis results using Plotly. It provides charts for throughput, latency,
# scalability analysis, and real-time monitoring.
# ============================================================================

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from typing import List, Dict, Any, Optional
import numpy as np
from datetime import datetime, timedelta

class Visualizer:
    """
    Visualization utilities for performance metrics and analysis results
    Creates interactive charts using Plotly
    """
    
    def __init__(self):
        self.color_palette = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'success': '#2ca02c',
            'danger': '#d62728',
            'warning': '#ff9800',
            'info': '#17a2b8'
        }
    
    def create_throughput_chart(self, performance_data: List[Dict[str, Any]]) -> go.Figure:
        """
        Create throughput visualization chart
        """
        if not performance_data or len(performance_data) == 0:
            return self._create_empty_chart("No performance data available")
        
        # Convert to DataFrame for easier handling
        df = pd.DataFrame(performance_data)
        
        if 'throughput' not in df.columns:
            fig = go.Figure()
            fig.add_annotation(
                text="No throughput data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        # Create throughput over time chart
        fig = go.Figure()
        
        if 'timestamp' in df.columns:
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['throughput'],
                mode='lines+markers',
                name='Throughput',
                line=dict(color=self.color_palette['primary'], width=2),
                marker=dict(size=6)
            ))
            
            fig.update_layout(
                title='Throughput Over Time',
                xaxis_title='Time',
                yaxis_title='Throughput (items/second)',
                hovermode='x unified'
            )
        else:
            # Bar chart for different operations
            if 'operation' in df.columns:
                fig.add_trace(go.Bar(
                    x=df['operation'],
                    y=df['throughput'],
                    name='Throughput',
                    marker_color=self.color_palette['primary']
                ))
            else:
                fig.add_trace(go.Bar(
                    x=list(range(len(df))),
                    y=df['throughput'],
                    name='Throughput',
                    marker_color=self.color_palette['primary']
                ))
            
            fig.update_layout(
                title='Throughput by Operation',
                xaxis_title='Operation',
                yaxis_title='Throughput (items/second)'
            )
        
        return fig
    
    def create_latency_chart(self, performance_data: List[Dict[str, Any]]) -> go.Figure:
        """
        Create latency visualization chart
        """
        if not performance_data:
            return self._create_empty_chart("No performance data available")
        
        df = pd.DataFrame(performance_data)
        
        if 'latency' not in df.columns and 'processing_time' not in df.columns:
            return self._create_empty_chart("No latency data available")
        
        # Use latency if available, otherwise use processing_time
        latency_col = 'latency' if 'latency' in df.columns else 'processing_time'
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Latency Over Time', 'Latency Distribution'),
            vertical_spacing=0.12
        )
        
        # Latency over time
        if 'timestamp' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df[latency_col],
                    mode='lines+markers',
                    name='Latency',
                    line=dict(color=self.color_palette['secondary'], width=2),
                    marker=dict(size=4)
                ),
                row=1, col=1
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(df))),
                    y=df[latency_col],
                    mode='lines+markers',
                    name='Latency',
                    line=dict(color=self.color_palette['secondary'], width=2),
                    marker=dict(size=4)
                ),
                row=1, col=1
            )
        
        # Latency distribution histogram
        fig.add_trace(
            go.Histogram(
                x=df[latency_col],
                name='Distribution',
                marker_color=self.color_palette['info'],
                opacity=0.7
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title='Latency Analysis',
            height=600,
            showlegend=False
        )
        
        fig.update_xaxes(title_text="Time", row=1, col=1)
        fig.update_yaxes(title_text="Latency (seconds)", row=1, col=1)
        fig.update_xaxes(title_text="Latency (seconds)", row=2, col=1)
        fig.update_yaxes(title_text="Frequency", row=2, col=1)
        
        return fig
    
    def create_scalability_chart(self, performance_data: List[Dict[str, Any]]) -> go.Figure:
        """
        Create scalability analysis chart
        """
        if not performance_data:
            return self._create_empty_chart("No performance data available")
        
        df = pd.DataFrame(performance_data)
        
        # Look for scaling-related columns
        if 'num_workers' not in df.columns:
            return self._create_empty_chart("No scalability data available")
        
        # Group by number of workers and calculate average throughput
        if 'throughput' in df.columns:
            scaling_data = df.groupby('num_workers')['throughput'].mean().reset_index()
        else:
            return self._create_empty_chart("No throughput data for scalability analysis")
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Throughput vs Workers', 'Scaling Efficiency'),
            horizontal_spacing=0.1
        )
        
        # Throughput vs workers
        fig.add_trace(
            go.Scatter(
                x=scaling_data['num_workers'],
                y=scaling_data['throughput'],
                mode='lines+markers',
                name='Actual Throughput',
                line=dict(color=self.color_palette['primary'], width=3),
                marker=dict(size=8)
            ),
            row=1, col=1
        )
        
        # Ideal linear scaling line
        if len(scaling_data) > 1:
            base_throughput = scaling_data.iloc[0]['throughput']
            base_workers = scaling_data.iloc[0]['num_workers']
            ideal_throughput = [base_throughput * (workers / base_workers) for workers in scaling_data['num_workers']]
            
            fig.add_trace(
                go.Scatter(
                    x=scaling_data['num_workers'],
                    y=ideal_throughput,
                    mode='lines',
                    name='Ideal Linear Scaling',
                    line=dict(color=self.color_palette['success'], width=2, dash='dash')
                ),
                row=1, col=1
            )
        
        # Calculate efficiency
        if len(scaling_data) > 1:
            base_throughput = scaling_data.iloc[0]['throughput']
            base_workers = scaling_data.iloc[0]['num_workers']
            efficiency = []
            
            for _, row in scaling_data.iterrows():
                speedup = row['throughput'] / base_throughput
                theoretical_speedup = row['num_workers'] / base_workers
                eff = speedup / theoretical_speedup if theoretical_speedup > 0 else 0
                efficiency.append(eff * 100)  # Convert to percentage
            
            fig.add_trace(
                go.Scatter(
                    x=scaling_data['num_workers'],
                    y=efficiency,
                    mode='lines+markers',
                    name='Efficiency',
                    line=dict(color=self.color_palette['warning'], width=3),
                    marker=dict(size=8)
                ),
                row=1, col=2
            )
        
        fig.update_layout(
            title='Scalability Analysis',
            height=400
        )
        
        fig.update_xaxes(title_text="Number of Workers", row=1, col=1)
        fig.update_yaxes(title_text="Throughput (items/sec)", row=1, col=1)
        fig.update_xaxes(title_text="Number of Workers", row=1, col=2)
        fig.update_yaxes(title_text="Efficiency (%)", row=1, col=2)
        
        return fig
    
    def create_performance_comparison_chart(self, sequential_time: float, 
                                          parallel_time: float, 
                                          num_workers: int) -> go.Figure:
        """
        Create performance comparison chart (sequential vs parallel)
        """
        categories = ['Sequential', 'Parallel']
        times = [sequential_time, parallel_time]
        speedup = sequential_time / parallel_time if parallel_time > 0 else 0
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(f'Processing Time Comparison', f'Speedup: {speedup:.2f}x'),
            specs=[[{"type": "bar"}, {"type": "indicator"}]]
        )
        
        # Bar chart comparing times
        colors = [self.color_palette['danger'], self.color_palette['success']]
        fig.add_trace(
            go.Bar(
                x=categories,
                y=times,
                name='Processing Time',
                marker_color=colors,
                text=[f'{t:.2f}s' for t in times],
                textposition='auto'
            ),
            row=1, col=1
        )
        
        # Speedup indicator
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=speedup,
                delta={'reference': 1, 'position': "top"},
                gauge={
                    'axis': {'range': [None, max(speedup * 1.2, num_workers)]},
                    'bar': {'color': self.color_palette['primary']},
                    'steps': [
                        {'range': [0, 1], 'color': "lightgray"},
                        {'range': [1, num_workers], 'color': "lightgreen"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': num_workers
                    }
                },
                title={'text': "Speedup Factor"}
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title=f'Sequential vs Parallel Performance ({num_workers} workers)',
            height=400,
            showlegend=False
        )
        
        fig.update_yaxes(title_text="Time (seconds)", row=1, col=1)
        
        return fig
    
    def create_word_frequency_chart(self, word_counts: Dict[str, int], top_n: int = 20) -> go.Figure:
        """
        Create word frequency visualization
        """
        if not word_counts:
            return self._create_empty_chart("No word frequency data available")
        
        # Get top N words
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
        words, counts = zip(*sorted_words)
        
        fig = go.Figure(data=[
            go.Bar(
                x=list(words),
                y=list(counts),
                marker_color=self.color_palette['primary'],
                text=list(counts),
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title=f'Top {top_n} Most Frequent Words',
            xaxis_title='Words',
            yaxis_title='Frequency',
            xaxis_tickangle=-45
        )
        
        return fig
    
    def create_sentiment_distribution_chart(self, sentiment_data: List[Dict[str, Any]]) -> go.Figure:
        """
        Create sentiment analysis visualization
        """
        if not sentiment_data:
            return self._create_empty_chart("No sentiment data available")
        
        df = pd.DataFrame(sentiment_data)
        
        if 'sentiment' not in df.columns:
            return self._create_empty_chart("No sentiment classification data available")
        
        # Count sentiment distribution
        sentiment_counts = df['sentiment'].value_counts()
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Sentiment Distribution', 'Polarity Scores'),
            specs=[[{"type": "pie"}, {"type": "histogram"}]]
        )
        
        # Pie chart for sentiment distribution
        colors = {
            'positive': self.color_palette['success'],
            'negative': self.color_palette['danger'],
            'neutral': self.color_palette['info']
        }
        
        fig.add_trace(
            go.Pie(
                labels=sentiment_counts.index,
                values=sentiment_counts.values,
                marker_colors=[colors.get(label, self.color_palette['primary']) for label in sentiment_counts.index]
            ),
            row=1, col=1
        )
        
        # Histogram for polarity scores
        if 'polarity' in df.columns:
            fig.add_trace(
                go.Histogram(
                    x=df['polarity'],
                    nbinsx=20,
                    marker_color=self.color_palette['secondary'],
                    opacity=0.7
                ),
                row=1, col=2
            )
        
        fig.update_layout(
            title='Sentiment Analysis Results',
            height=400,
            showlegend=False
        )
        
        return fig
    
    def create_real_time_metrics_chart(self, metrics_data: List[Dict[str, Any]]) -> go.Figure:
        """
        Create real-time metrics dashboard
        """
        if not metrics_data:
            return self._create_empty_chart("No real-time metrics available")
        
        df = pd.DataFrame(metrics_data)
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('CPU Usage', 'Memory Usage', 'Processing Rate', 'Queue Size'),
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )
        
        # CPU Usage
        if 'cpu_percent' in df.columns and 'timestamp' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['cpu_percent'],
                    mode='lines',
                    name='CPU %',
                    line=dict(color=self.color_palette['primary'])
                ),
                row=1, col=1
            )
        
        # Memory Usage
        if 'memory_percent' in df.columns and 'timestamp' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['memory_percent'],
                    mode='lines',
                    name='Memory %',
                    line=dict(color=self.color_palette['secondary'])
                ),
                row=1, col=2
            )
        
        # Processing Rate
        if 'throughput' in df.columns and 'timestamp' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['throughput'],
                    mode='lines',
                    name='Throughput',
                    line=dict(color=self.color_palette['success'])
                ),
                row=2, col=1
            )
        
        # Queue Size (if available)
        if 'queue_size' in df.columns and 'timestamp' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['queue_size'],
                    mode='lines',
                    name='Queue Size',
                    line=dict(color=self.color_palette['warning'])
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            title='Real-time System Metrics',
            height=600,
            showlegend=False
        )
        
        return fig
    
    def _create_empty_chart(self, message: str) -> go.Figure:
        """
        Create an empty chart with a message
        """
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            plot_bgcolor='white'
        )
        return fig
