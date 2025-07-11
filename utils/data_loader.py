# ============================================================================
# Data Loader - Implemented by: Abhijith
# 
# This module handles data ingestion from various file formats and provides
# sample datasets for testing. It supports CSV, JSON, and text files.
# ============================================================================

import pandas as pd
import json
import csv
from typing import List, Dict, Any, Union
import io
import os
from pathlib import Path

class DataLoader:
    """
    Data loading utilities for various text formats
    Handles file uploads and sample datasets
    """
    
    def __init__(self):
        self.supported_formats = ['txt', 'csv', 'json']
    
    def load_file(self, uploaded_file) -> List[str]:
        """
        Load text data from uploaded file
        """
        try:
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            if file_extension == 'txt':
                return self._load_txt_file(uploaded_file)
            elif file_extension == 'csv':
                return self._load_csv_file(uploaded_file)
            elif file_extension == 'json':
                return self._load_json_file(uploaded_file)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
                
        except Exception as e:
            print(f"Error loading file {uploaded_file.name}: {str(e)}")
            return []
    
    def _load_txt_file(self, uploaded_file) -> List[str]:
        """
        Load text file - each line becomes a document
        """
        content = uploaded_file.read().decode('utf-8')
        lines = content.strip().split('\n')
        
        # Filter out empty lines and very short lines
        documents = [line.strip() for line in lines if len(line.strip()) > 10]
        
        return documents
    
    def _load_csv_file(self, uploaded_file) -> List[str]:
        """
        Load CSV file - assumes text content in first text column found
        """
        try:
            df = pd.read_csv(uploaded_file)
            
            # Find the first column that likely contains text
            text_column = None
            for col in df.columns:
                if df[col].dtype == 'object':  # String/text column
                    # Check if column contains substantial text
                    sample_text = str(df[col].iloc[0]) if len(df) > 0 else ""
                    if len(sample_text) > 20:  # Likely contains text content
                        text_column = col
                        break
            
            if text_column is None:
                # Fallback to first column
                text_column = df.columns[0]
            
            # Extract text data
            documents = []
            for _, row in df.iterrows():
                text = str(row[text_column]).strip()
                if len(text) > 10:  # Filter out very short texts
                    documents.append(text)
            
            return documents
            
        except Exception as e:
            # Return empty list instead of crashing
            return []
    
    def _load_json_file(self, uploaded_file) -> List[str]:
        """
        Load JSON file - handles various JSON structures
        """
        try:
            content = uploaded_file.read().decode('utf-8')
            data = json.loads(content)
            
            documents = []
            
            if isinstance(data, list):
                # Array of objects or strings
                for item in data:
                    if isinstance(item, str):
                        if len(item.strip()) > 10:
                            documents.append(item.strip())
                    elif isinstance(item, dict):
                        # Extract text from common fields
                        text = self._extract_text_from_dict(item)
                        if text and len(text) > 10:
                            documents.append(text)
            
            elif isinstance(data, dict):
                # Single object - extract text fields
                text = self._extract_text_from_dict(data)
                if text and len(text) > 10:
                    documents.append(text)
            
            return documents
            
        except Exception as e:
            # Return empty list instead of crashing
            return []
    
    def _extract_text_from_dict(self, item: Dict) -> str:
        """
        Extract text content from dictionary object
        """
        # Common text field names
        text_fields = ['text', 'content', 'message', 'description', 'body', 
                      'comment', 'review', 'tweet', 'post', 'title']
        
        for field in text_fields:
            if field in item and isinstance(item[field], str):
                return item[field].strip()
        
        # If no common fields found, concatenate all string values
        text_parts = []
        for value in item.values():
            if isinstance(value, str) and len(value.strip()) > 5:
                text_parts.append(value.strip())
        
        return ' '.join(text_parts) if text_parts else ""
    
    def load_sample_text(self) -> List[str]:
        """
        Load sample text file for demonstration
        """
        sample_texts = [
            "This is an excellent example of parallel computing in action. The performance improvements are remarkable when using multiple processors.",
            "Machine learning algorithms benefit greatly from distributed processing. The scalability is amazing and results are outstanding.",
            "Cloud computing has revolutionized how we process big data. AWS services provide incredible infrastructure for parallel processing.",
            "The implementation of MapReduce is fantastic for handling large datasets. It's a wonderful approach to distributed computing.",
            "Stream processing enables real-time analytics which is absolutely essential for modern applications. The throughput is incredible.",
            "Sentiment analysis using parallel processing shows tremendous improvements in performance and accuracy.",
            "The benchmarking results demonstrate the superiority of parallel algorithms over sequential processing methods.",
            "This text processing system is poorly designed and shows terrible performance under load. The results are disappointing.",
            "The user interface is confusing and the documentation is inadequate. Very frustrating experience overall.",
            "Data ingestion capabilities are limited and the error handling is awful. Not satisfied with the implementation.",
            "Parallel computing concepts are fundamental to modern data processing. Understanding MapReduce is crucial for big data.",
            "The sliding window operations in stream processing provide excellent real-time insights into data patterns.",
            "Performance monitoring and benchmarking are essential components of any distributed system architecture.",
            "Text analytics using natural language processing techniques can extract valuable insights from unstructured data.",
            "The visualization dashboard provides comprehensive metrics and graphs for performance analysis and optimization.",
            "Multiprocessing in Python enables efficient utilization of multiple CPU cores for computational tasks.",
            "The system architecture supports both batch processing and real-time stream processing for different use cases.",
            "Auto-scaling policies ensure optimal resource utilization based on workload demands and performance requirements.",
            "Data visualization helps in understanding complex patterns and trends in the processed information.",
            "The integration of various AWS services creates a robust and scalable cloud-based processing pipeline."
        ]
        
        return sample_texts
    
    def load_sample_tweets(self) -> List[str]:
        """
        Load sample tweet-like data for demonstration
        """
        sample_tweets = [
            "Just deployed my first #MapReduce job on AWS! The parallel processing is incredible 🚀 #BigData #CloudComputing",
            "Learning about distributed systems and loving every minute of it! #TechLife #Programming #Python",
            "Stream processing with sliding windows is absolutely amazing for real-time analytics! 📊 #DataScience",
            "Sentiment analysis on social media data reveals fascinating insights about public opinion trends 🤔",
            "The performance improvements from parallel computing are mind-blowing! Sequential vs parallel = night and day ⚡",
            "Working on text processing algorithms. The throughput gains from multiprocessing are outstanding! 💪",
            "Cloud computing has changed everything. AWS makes scaling so much easier! ☁️ #CloudFirst",
            "Frustrated with slow data processing. Need better algorithms and more efficient implementations 😤",
            "This machine learning course is terrible. The explanations are confusing and examples don't work 😠",
            "Disappointed with the performance of this new processing framework. Expected much better results 👎",
            "Real-time data processing is the future! Stream analytics enables instant decision making 🔥",
            "Benchmarking different algorithms to find the optimal solution for our use case 📈 #PerformanceOptimization",
            "The scalability of modern distributed systems continues to amaze me every day! 🌐",
            "Text mining and NLP are opening up incredible possibilities for automated content analysis 🤖",
            "Visualizing performance metrics helps identify bottlenecks and optimization opportunities 📊",
            "Python's multiprocessing library makes parallel computing accessible to everyone! 🐍 #PythonProgramming",
            "Hybrid parallelism combining data and task parallelism delivers exceptional performance gains ⚡",
            "Auto-scaling in the cloud ensures we never run out of compute resources during peak loads 📈",
            "The future of data processing is distributed, parallel, and cloud-native! 🚀 #FutureOfTech",
            "Amazing how sliding window operations can provide real-time insights into streaming data patterns! 🌊"
        ]
        
        return sample_tweets
    
    def validate_data_format(self, data: List[str]) -> Dict[str, Any]:
        """
        Validate and analyze loaded data
        """
        if not data:
            return {'valid': False, 'error': 'No data loaded'}
        
        analysis = {
            'valid': True,
            'total_documents': len(data),
            'empty_documents': len([d for d in data if not d.strip()]),
            'short_documents': len([d for d in data if len(d.split()) < 5]),
            'average_length': sum(len(doc.split()) for doc in data) / len(data),
            'total_words': sum(len(doc.split()) for doc in data),
            'unique_words': len(set(' '.join(data).lower().split())),
        }
        
        # Calculate statistics with error handling
        try:
            doc_lengths = [len(doc.split()) for doc in data if doc]
            analysis['min_length'] = min(doc_lengths) if doc_lengths else 0
            analysis['max_length'] = max(doc_lengths) if doc_lengths else 0
        except Exception:
            analysis['min_length'] = 0
            analysis['max_length'] = 0
        
        # Quality assessment
        if analysis['empty_documents'] > len(data) * 0.2:  # More than 20% empty
            analysis['warnings'] = analysis.get('warnings', [])
            analysis['warnings'].append('High percentage of empty documents')
        
        if analysis['short_documents'] > len(data) * 0.3:  # More than 30% very short
            analysis['warnings'] = analysis.get('warnings', [])
            analysis['warnings'].append('High percentage of very short documents')
        
        return analysis
    
    def create_sample_dataset(self, size: int = 1000, theme: str = "technology") -> List[str]:
        """
        Generate a sample dataset of specified size and theme
        """
        base_texts = {
            "technology": [
                "Artificial intelligence is transforming industries across the globe with machine learning algorithms",
                "Cloud computing platforms provide scalable infrastructure for modern applications and services",
                "Data science and analytics help organizations make informed decisions based on empirical evidence",
                "Cybersecurity measures are essential to protect sensitive information from malicious attacks",
                "Internet of Things devices are creating interconnected ecosystems of smart technology",
                "Blockchain technology offers decentralized solutions for secure and transparent transactions",
                "Quantum computing promises to solve complex problems that are intractable for classical computers",
                "Virtual reality and augmented reality are changing how we interact with digital content",
                "5G networks enable faster communication and support for next-generation mobile applications",
                "Edge computing brings processing power closer to data sources for reduced latency"
            ],
            "business": [
                "Strategic planning involves setting long-term goals and developing actionable implementation strategies",
                "Market research provides valuable insights into customer preferences and competitive landscapes",
                "Financial analysis helps organizations optimize resource allocation and improve profitability",
                "Supply chain management ensures efficient flow of goods from suppliers to customers",
                "Human resources development focuses on talent acquisition and employee engagement strategies",
                "Digital transformation initiatives modernize business processes using advanced technologies",
                "Customer relationship management systems improve service quality and retention rates",
                "Project management methodologies ensure timely delivery of objectives within budget constraints",
                "Risk assessment procedures identify potential threats and develop mitigation strategies",
                "Performance metrics and key performance indicators measure organizational success"
            ]
        }
        
        theme_texts = base_texts.get(theme, base_texts["technology"])
        generated_texts = []
        
        for i in range(size):
            # Select base text and add variations
            base_text = theme_texts[i % len(theme_texts)]
            
            # Add some variation
            variations = [
                f"In today's world, {base_text.lower()}",
                f"Research shows that {base_text.lower()}",
                f"Many experts believe that {base_text.lower()}",
                f"Studies indicate that {base_text.lower()}",
                f"It is widely recognized that {base_text.lower()}",
                base_text,
                f"According to recent findings, {base_text.lower()}",
                f"The latest developments show that {base_text.lower()}"
            ]
            
            selected_text = variations[i % len(variations)]
            generated_texts.append(selected_text)
        
        return generated_texts
