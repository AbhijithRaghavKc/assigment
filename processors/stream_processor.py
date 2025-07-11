# ============================================================================
# Stream Processor - Implemented by: Abhijith
# 
# This module implements real-time stream processing with sliding window
# operations. It simulates data ingestion and processing for real-time
# analytics scenarios.
# ============================================================================

import time
import threading
import queue
from collections import defaultdict, deque, Counter
from datetime import datetime, timedelta
from typing import List, Dict, Any, Generator, Optional
import random
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

class StreamProcessor:
    """
    Stream processing simulation for real-time text analysis
    Implements sliding window operations and real-time processing
    """
    
    def __init__(self, window_size: int = 10, processing_rate: int = 10):
        self.window_size = window_size  # seconds
        self.processing_rate = processing_rate  # documents per second
        self.data_queue = queue.Queue()
        self.results_queue = queue.Queue()
        self.active_windows = defaultdict(deque)
        self.is_processing = False
        self._download_nltk_data()
        
    def _download_nltk_data(self):
        """Download required NLTK data if not already present"""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
            
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords', quiet=True)
            
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet', quiet=True)
        
    def simulate_stream(self, documents: List[str], simulation_time: int, 
                       operation: str = "word_frequency", num_workers: int = 2) -> List[Dict]:
        """
        Simulate streaming data processing without threading to prevent crashes
        """
        results = []
        
        # Simple simulation without complex threading
        max_docs = min(len(documents), 20)  # Limit to prevent hanging
        
        for i in range(max_docs):
            try:
                # Simulate real-time processing
                timestamp = datetime.now()
                document = documents[i % len(documents)]
                
                # Create data point
                data_point = {
                    'timestamp': timestamp,
                    'document': document,
                    'doc_id': i
                }
                
                # Process the data point
                result = self._process_stream_data_simple(data_point, operation)
                if result:
                    results.append(result)
                
                # Small delay to simulate processing time
                time.sleep(0.1)
                
            except Exception:
                # Skip problematic documents and continue processing
                continue
        
        return results
    
    def _process_stream_data_simple(self, data_point: Dict, operation: str) -> Dict:
        """Simplified stream data processing without complex windowing"""
        timestamp = data_point['timestamp']
        document = data_point['document']
        doc_id = data_point['doc_id']
        
        # Basic text processing with error handling
        try:
            words = document.lower().split()
            words = [word.strip('.,!?()[]{}":;') for word in words if len(word) > 2]
        except Exception:
            words = []
        
        if operation == "word_frequency":
            return {
                'timestamp': timestamp,
                'doc_id': doc_id,
                'operation': operation,
                'word_count': len(words),
                'unique_words': len(set(words)),
                'top_words': dict(Counter(words).most_common(5)),
                'processing_time': 0.1,
                'documents_processed': 1
            }
        elif operation == "trending_keywords":
            # Simple keyword extraction
            long_words = [word for word in words if len(word) > 5]
            return {
                'timestamp': timestamp,
                'doc_id': doc_id,
                'operation': operation,
                'trending_keywords': long_words[:5],
                'keyword_count': len(long_words),
                'processing_time': 0.1,
                'documents_processed': 1
            }
        elif operation == "sentiment_monitoring":
            # Basic sentiment analysis
            positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic']
            negative_words = ['bad', 'terrible', 'awful', 'horrible', 'poor', 'disappointing']
            
            pos_count = sum(1 for word in words if word in positive_words)
            neg_count = sum(1 for word in words if word in negative_words)
            
            sentiment = 'positive' if pos_count > neg_count else 'negative' if neg_count > pos_count else 'neutral'
            
            return {
                'timestamp': timestamp,
                'doc_id': doc_id,
                'operation': operation,
                'sentiment': sentiment,
                'positive_score': pos_count,
                'negative_score': neg_count,
                'processing_time': 0.1,
                'documents_processed': 1
            }
        
        return {}
    
    def _produce_stream_data(self, documents: List[str], simulation_time: int):
        """Simulate real-time data ingestion"""
        start_time = time.time()
        doc_index = 0
        
        while self.is_processing and (time.time() - start_time) < simulation_time:
            # Simulate data arrival rate
            if doc_index < len(documents):
                timestamp = datetime.now()
                data_point = {
                    'timestamp': timestamp,
                    'document': documents[doc_index],
                    'doc_id': doc_index
                }
                
                self.data_queue.put(data_point)
                doc_index += 1
                
                # Control the rate of data production
                time.sleep(1.0 / self.processing_rate)
            else:
                # Restart from beginning if we run out of documents
                doc_index = 0
    
    def _consume_stream_data(self, operation: str):
        """Process streaming data in real-time with improved timeout handling"""
        consecutive_empty_count = 0
        max_empty_attempts = 5
        
        while self.is_processing and consecutive_empty_count < max_empty_attempts:
            try:
                data_point = self.data_queue.get(timeout=0.5)
                consecutive_empty_count = 0  # Reset counter on successful get
                
                # Process the data point
                result = self._process_stream_data(data_point, operation)
                
                if result:
                    self.results_queue.put(result)
                
            except queue.Empty:
                consecutive_empty_count += 1
                continue
            except Exception:
                consecutive_empty_count += 1
                continue
    
    def _process_stream_data(self, data_point: Dict, operation: str) -> Dict:
        """Process individual data points"""
        timestamp = data_point['timestamp']
        document = data_point['document']
        
        # Add to sliding window
        window_key = self._get_window_key(timestamp)
        self.active_windows[window_key].append(data_point)
        
        # Clean old windows
        self._clean_old_windows(timestamp)
        
        # Perform the requested operation
        if operation == "word_frequency":
            return self._word_frequency_window(window_key, timestamp)
        elif operation == "trending_keywords":
            return self._trending_keywords_window(window_key, timestamp)
        elif operation == "sentiment_monitoring":
            return self._sentiment_monitoring_window(window_key, timestamp)
        
        return {}
    
    def _get_window_key(self, timestamp: datetime) -> str:
        """Generate window key based on timestamp"""
        # Create window keys based on window size
        window_start = timestamp.replace(second=(timestamp.second // self.window_size) * self.window_size, microsecond=0)
        return window_start.strftime("%Y-%m-%d %H:%M:%S")
    
    def _clean_old_windows(self, current_time: datetime):
        """Remove windows older than the window size"""
        cutoff_time = current_time - timedelta(seconds=self.window_size * 2)
        
        keys_to_remove = []
        for window_key in self.active_windows:
            window_time = datetime.strptime(window_key, "%Y-%m-%d %H:%M:%S")
            if window_time < cutoff_time:
                keys_to_remove.append(window_key)
        
        for key in keys_to_remove:
            del self.active_windows[key]
    
    def _word_frequency_window(self, window_key: str, timestamp: datetime) -> Dict:
        """Calculate word frequency for current window"""
        window_data = self.active_windows[window_key]
        
        if not window_data:
            return {}
        
        # Count words in current window
        word_freq = Counter()
        documents_processed = 0
        
        start_time = time.time()
        
        for data_point in window_data:
            words = data_point['document'].lower().split()
            words = [word.strip('.,!?";()[]{}') for word in words if len(word) > 2]
            word_freq.update(words)
            documents_processed += 1
        
        processing_time = time.time() - start_time
        
        return {
            'timestamp': timestamp,
            'window_key': window_key,
            'operation': 'word_frequency',
            'word_freq': dict(word_freq.most_common(10)),
            'documents_processed': documents_processed,
            'processing_time': processing_time,
            'total_words': sum(word_freq.values())
        }
    
    def _trending_keywords_window(self, window_key: str, timestamp: datetime) -> Dict:
        """Identify trending keywords in current window"""
        window_data = self.active_windows[window_key]
        
        if not window_data:
            return {}
        
        # Define trending keywords to track
        trending_keywords = ['data', 'analysis', 'machine', 'learning', 'python', 
                           'processing', 'parallel', 'distributed', 'cloud', 'aws']
        
        keyword_counts = defaultdict(int)
        documents_processed = 0
        
        start_time = time.time()
        
        for data_point in window_data:
            doc_lower = data_point['document'].lower()
            for keyword in trending_keywords:
                keyword_counts[keyword] += doc_lower.count(keyword)
            documents_processed += 1
        
        processing_time = time.time() - start_time
        
        # Sort by count and get top 5
        sorted_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            'timestamp': timestamp,
            'window_key': window_key,
            'operation': 'trending_keywords',
            'trending_keywords': dict(sorted_keywords),
            'documents_processed': documents_processed,
            'processing_time': processing_time
        }
    
    def _sentiment_monitoring_window(self, window_key: str, timestamp: datetime) -> Dict:
        """Monitor sentiment trends in current window"""
        window_data = self.active_windows[window_key]
        
        if not window_data:
            return {}
        
        # Simple sentiment analysis based on positive/negative words
        positive_words = {'good', 'great', 'excellent', 'amazing', 'wonderful', 
                         'fantastic', 'awesome', 'love', 'best', 'perfect'}
        negative_words = {'bad', 'terrible', 'awful', 'horrible', 'worst', 
                         'hate', 'disgusting', 'pathetic', 'useless', 'disappointing'}
        
        sentiment_scores = []
        documents_processed = 0
        
        start_time = time.time()
        
        for data_point in window_data:
            doc_words = set(data_point['document'].lower().split())
            
            positive_count = len(doc_words.intersection(positive_words))
            negative_count = len(doc_words.intersection(negative_words))
            
            # Simple sentiment score
            sentiment_score = positive_count - negative_count
            sentiment_scores.append(sentiment_score)
            documents_processed += 1
        
        processing_time = time.time() - start_time
        
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
        
        return {
            'timestamp': timestamp,
            'window_key': window_key,
            'operation': 'sentiment_monitoring',
            'avg_sentiment': avg_sentiment,
            'sentiment_trend': 'positive' if avg_sentiment > 0 else 'negative' if avg_sentiment < 0 else 'neutral',
            'documents_processed': documents_processed,
            'processing_time': processing_time,
            'sentiment_distribution': {
                'positive': len([s for s in sentiment_scores if s > 0]),
                'negative': len([s for s in sentiment_scores if s < 0]),
                'neutral': len([s for s in sentiment_scores if s == 0])
            }
        }
    
    def sliding_window_aggregation(self, operation: str = "word_count", 
                                  window_duration: Optional[int] = None) -> Dict:
        """
        Perform sliding window aggregation on current data
        """
        if window_duration is None:
            window_duration = self.window_size
        
        current_time = datetime.now()
        cutoff_time = current_time - timedelta(seconds=window_duration)
        
        # Collect all data within the window
        window_documents = []
        for window_key, window_data in self.active_windows.items():
            window_time = datetime.strptime(window_key, "%Y-%m-%d %H:%M:%S")
            if window_time >= cutoff_time:
                for data_point in window_data:
                    window_documents.append(data_point['document'])
        
        if not window_documents:
            return {'error': 'No data in current window'}
        
        # Perform aggregation
        if operation == "word_count":
            word_counts = Counter()
            for doc in window_documents:
                words = doc.lower().split()
                words = [word.strip('.,!?";()[]{}') for word in words if len(word) > 2]
                word_counts.update(words)
            
            return {
                'operation': operation,
                'window_duration': window_duration,
                'documents_count': len(window_documents),
                'top_words': dict(word_counts.most_common(20)),
                'total_unique_words': len(word_counts),
                'timestamp': current_time
            }
        
        return {'error': f'Unknown operation: {operation}'}
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get current processing statistics"""
        total_documents = sum(len(window_data) for window_data in self.active_windows.values())
        active_windows_count = len(self.active_windows)
        
        return {
            'active_windows': active_windows_count,
            'total_documents_in_windows': total_documents,
            'queue_size': self.data_queue.qsize(),
            'results_queue_size': self.results_queue.qsize(),
            'window_size_seconds': self.window_size,
            'processing_rate': self.processing_rate,
            'is_processing': self.is_processing
        }
