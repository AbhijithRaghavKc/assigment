# ============================================================================
# Sentiment Analyzer - Implemented by: Anurag
# 
# This module provides parallel sentiment analysis capabilities using
# rule-based lexicon approach. It demonstrates parallel processing for
# natural language processing tasks.
# ============================================================================

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from typing import List, Dict, Any, Tuple, Optional
import re
from collections import defaultdict
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer

class SentimentAnalyzer:
    """
    Sentiment analysis processor with parallel execution capabilities
    Uses rule-based sentiment analysis for demonstration
    """
    
    def __init__(self):
        # Predefined sentiment lexicons for rule-based analysis
        # Enhanced with NLTK preprocessing - Implemented by: Anurag
        self.stemmer = PorterStemmer()
        self._download_nltk_data()
        
        self.positive_words = {
            'excellent', 'amazing', 'wonderful', 'fantastic', 'great', 'good', 
            'awesome', 'outstanding', 'brilliant', 'superb', 'perfect', 'love',
            'best', 'incredible', 'magnificent', 'marvelous', 'exceptional',
            'remarkable', 'terrific', 'splendid', 'delightful', 'pleased',
            'happy', 'satisfied', 'excited', 'thrilled', 'impressed', 'glad'
        }
        
        self.negative_words = {
            'terrible', 'awful', 'horrible', 'bad', 'worst', 'hate', 'disgusting',
            'pathetic', 'useless', 'disappointing', 'poor', 'dreadful', 'appalling',
            'atrocious', 'abysmal', 'deplorable', 'disastrous', 'inadequate',
            'inferior', 'substandard', 'unsatisfactory', 'annoyed', 'frustrated',
            'angry', 'upset', 'displeased', 'irritated', 'sad', 'depressed'
        }
        
        # Intensifiers and negation words
        self.intensifiers = {'very', 'extremely', 'incredibly', 'absolutely', 'totally'}
        self.negation_words = {'not', 'never', 'no', 'nothing', 'neither', 'nowhere', 'none'}
    
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
    
    def analyze_parallel(self, documents: List[str], num_workers: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Perform parallel sentiment analysis on documents
        """
        if num_workers is None:
            num_workers = mp.cpu_count()
        
        # Split documents into chunks for parallel processing with error handling
        if len(documents) == 0:
            return []
        
        chunk_size = max(1, len(documents) // num_workers)
        chunks = [documents[i:i + chunk_size] for i in range(0, len(documents), chunk_size)]
        
        # Process chunks in parallel with error handling
        try:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                future_to_chunk = {
                    executor.submit(self._analyze_chunk, chunk, chunk_idx): chunk_idx 
                    for chunk_idx, chunk in enumerate(chunks)
                }
                
                results = []
                for future in as_completed(future_to_chunk):
                    try:
                        chunk_results = future.result()
                        results.extend(chunk_results)
                    except Exception:
                        # Skip failed chunks
                        continue
        except Exception:
            # Fallback to sequential processing if parallel fails
            results = []
            for chunk_idx, chunk in enumerate(chunks):
                try:
                    chunk_results = self._analyze_chunk(chunk, chunk_idx)
                    results.extend(chunk_results)
                except Exception:
                    continue
        
        return results
    
    def analyze_sequential(self, documents: List[str]) -> List[Dict[str, Any]]:
        """
        Perform sequential sentiment analysis for comparison
        """
        results = []
        for idx, document in enumerate(documents):
            sentiment_result = self._analyze_document(document, idx)
            results.append(sentiment_result)
        
        return results
    
    def _analyze_chunk(self, chunk: List[str], chunk_idx: int) -> List[Dict[str, Any]]:
        """
        Analyze a chunk of documents
        """
        results = []
        for doc_idx, document in enumerate(chunk):
            global_idx = chunk_idx * len(chunk) + doc_idx
            sentiment_result = self._analyze_document(document, global_idx)
            results.append(sentiment_result)
        
        return results
    
    def _analyze_document(self, document: str, doc_id: int) -> Dict[str, Any]:
        """
        Analyze sentiment of a single document
        """
        # Preprocess text
        text = self._preprocess_text(document)
        words = text.split()
        
        # Calculate sentiment scores
        sentiment_scores = self._calculate_sentiment_scores(words)
        
        # Determine overall sentiment
        polarity = sentiment_scores['polarity']
        
        if polarity > 0.1:
            sentiment = 'positive'
        elif polarity < -0.1:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        return {
            'doc_id': doc_id,
            'text': document[:100] + '...' if len(document) > 100 else document,
            'sentiment': sentiment,
            'polarity': polarity,
            'positive_score': sentiment_scores['positive_score'],
            'negative_score': sentiment_scores['negative_score'],
            'word_count': len(words),
            'sentiment_words_found': sentiment_scores['sentiment_words']
        }
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text for sentiment analysis using NLTK
        Enhanced preprocessing - Implemented by: Anurag
        """
        try:
            # Use NLTK sentence tokenization first
            sentences = sent_tokenize(text)
            processed_sentences = []
            
            for sentence in sentences:
                # Tokenize words
                tokens = word_tokenize(sentence.lower())
                
                # Remove stopwords but keep sentiment-bearing words
                stop_words = set(stopwords.words('english'))
                # Keep negation and intensifier words even if they're stopwords
                filtered_tokens = [
                    token for token in tokens 
                    if token.isalpha() and (
                        token not in stop_words or 
                        token in self.negation_words or 
                        token in self.intensifiers
                    )
                ]
                
                # Apply stemming to normalize word forms
                stemmed_tokens = [self.stemmer.stem(token) for token in filtered_tokens]
                processed_sentences.append(' '.join(stemmed_tokens))
            
            return ' '.join(processed_sentences)
            
        except Exception:
            # Fallback to basic preprocessing if NLTK fails
            text = text.lower()
            text = re.sub(r'[^\w\s\.\!\?]', ' ', text)
            text = ' '.join(text.split())
            return text
    
    def _calculate_sentiment_scores(self, words: List[str]) -> Dict[str, Any]:
        """
        Calculate sentiment scores based on word analysis
        """
        positive_score = 0
        negative_score = 0
        sentiment_words = []
        
        i = 0
        while i < len(words):
            word = words[i]
            
            # Check for intensifiers
            intensifier_multiplier = 1.0
            if i > 0 and words[i-1] in self.intensifiers:
                intensifier_multiplier = 1.5
            
            # Check for negation
            negation_active = False
            if i > 0 and words[i-1] in self.negation_words:
                negation_active = True
            elif i > 1 and words[i-2] in self.negation_words:
                negation_active = True
            
            # Calculate sentiment contribution
            if word in self.positive_words:
                score = 1.0 * intensifier_multiplier
                if negation_active:
                    negative_score += score
                    sentiment_words.append(f"NOT_{word}")
                else:
                    positive_score += score
                    sentiment_words.append(word)
            
            elif word in self.negative_words:
                score = 1.0 * intensifier_multiplier
                if negation_active:
                    positive_score += score
                    sentiment_words.append(f"NOT_{word}")
                else:
                    negative_score += score
                    sentiment_words.append(word)
            
            i += 1
        
        # Calculate polarity (-1 to 1)
        total_sentiment_words = positive_score + negative_score
        if total_sentiment_words > 0:
            polarity = (positive_score - negative_score) / total_sentiment_words
        else:
            polarity = 0.0
        
        return {
            'positive_score': positive_score,
            'negative_score': negative_score,
            'polarity': polarity,
            'sentiment_words': sentiment_words
        }
    
    def batch_sentiment_analysis(self, documents: List[str], batch_size: int = 100, 
                                num_workers: Optional[int] = None) -> Dict[str, Any]:
        """
        Process documents in batches with performance monitoring
        """
        if num_workers is None:
            num_workers = mp.cpu_count()
        
        start_time = time.time()
        
        # Process in batches
        all_results = []
        batch_processing_times = []
        
        for i in range(0, len(documents), batch_size):
            batch_start_time = time.time()
            batch = documents[i:i + batch_size]
            
            batch_results = self.analyze_parallel(batch, num_workers)
            all_results.extend(batch_results)
            
            batch_time = time.time() - batch_start_time
            batch_processing_times.append(batch_time)
        
        total_time = time.time() - start_time
        
        # Calculate statistics
        sentiment_distribution = defaultdict(int)
        polarity_scores = []
        
        for result in all_results:
            sentiment_distribution[result['sentiment']] += 1
            polarity_scores.append(result['polarity'])
        
        avg_polarity = sum(polarity_scores) / len(polarity_scores) if polarity_scores else 0
        
        return {
            'total_documents': len(documents),
            'processing_time': total_time,
            'throughput': len(documents) / total_time,
            'batch_size': batch_size,
            'num_workers': num_workers,
            'sentiment_distribution': dict(sentiment_distribution),
            'average_polarity': avg_polarity,
            'batch_processing_times': batch_processing_times,
            'results': all_results
        }
    
    def real_time_sentiment_stream(self, text_stream: List[str], 
                                  window_size: int = 10) -> List[Dict[str, Any]]:
        """
        Simulate real-time sentiment analysis with sliding window
        """
        results = []
        window_sentiments = []
        
        for i, text in enumerate(text_stream):
            # Analyze current text
            sentiment_result = self._analyze_document(text, i)
            
            # Add to sliding window
            window_sentiments.append(sentiment_result['polarity'])
            
            # Maintain window size
            if len(window_sentiments) > window_size:
                window_sentiments.pop(0)
            
            # Calculate window statistics
            window_avg = sum(window_sentiments) / len(window_sentiments)
            window_trend = self._calculate_trend(window_sentiments)
            
            result = {
                'timestamp': i,
                'text': text[:50] + '...' if len(text) > 50 else text,
                'sentiment': sentiment_result['sentiment'],
                'polarity': sentiment_result['polarity'],
                'window_average': window_avg,
                'window_trend': window_trend,
                'window_size': len(window_sentiments)
            }
            
            results.append(result)
        
        return results
    
    def _calculate_trend(self, sentiment_window: List[float]) -> str:
        """
        Calculate sentiment trend in the window
        """
        if len(sentiment_window) < 2:
            return 'stable'
        
        # Simple trend calculation based on first and second half
        mid_point = len(sentiment_window) // 2
        first_half_avg = sum(sentiment_window[:mid_point]) / mid_point if mid_point > 0 else 0
        second_half_avg = sum(sentiment_window[mid_point:]) / (len(sentiment_window) - mid_point)
        
        difference = second_half_avg - first_half_avg
        
        if difference > 0.1:
            return 'improving'
        elif difference < -0.1:
            return 'declining'
        else:
            return 'stable'
    
    def compare_processing_methods(self, documents: List[str]) -> Dict[str, Any]:
        """
        Compare parallel vs sequential processing performance
        """
        # Sequential processing
        start_time = time.time()
        sequential_results = self.analyze_sequential(documents)
        sequential_time = time.time() - start_time
        
        # Parallel processing
        start_time = time.time()
        parallel_results = self.analyze_parallel(documents)
        parallel_time = time.time() - start_time
        
        # Calculate speedup
        speedup = sequential_time / parallel_time if parallel_time > 0 else 0
        efficiency = speedup / mp.cpu_count()
        
        return {
            'sequential_time': sequential_time,
            'parallel_time': parallel_time,
            'speedup': speedup,
            'efficiency': efficiency,
            'documents_processed': len(documents),
            'sequential_throughput': len(documents) / sequential_time,
            'parallel_throughput': len(documents) / parallel_time,
            'cpu_cores': mp.cpu_count()
        }
