# ============================================================================
# MapReduce Processor - Implemented by: Anurag
# 
# This module implements the MapReduce programming paradigm for parallel
# text processing. It demonstrates how to split large datasets into chunks,
# process them in parallel, and combine the results.
# ============================================================================

import multiprocessing as mp
from collections import Counter, defaultdict
import time
import re
from functools import reduce
from typing import List, Dict, Any, Tuple, Optional
import concurrent.futures
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

class MapReduceProcessor:
    """
    MapReduce implementation for text processing tasks
    Demonstrates parallel computing concepts using Python multiprocessing
    Enhanced with NLTK for advanced text processing - Implemented by: Anurag
    """
    
    def __init__(self, num_workers: Optional[int] = None):
        self.num_workers = num_workers or mp.cpu_count()
        self.stemmer = PorterStemmer()
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
        
    def word_count_parallel(self, documents: List[str], chunk_size: int = 100) -> Counter:
        """
        Parallel word count using MapReduce pattern
        """
        # Split documents into chunks
        chunks = self._chunk_documents(documents, chunk_size)
        
        # Map phase: parallel word counting
        with mp.Pool(self.num_workers) as pool:
            map_results = pool.map(self._map_word_count, chunks)
        
        # Reduce phase: combine results
        final_counts = self._reduce_word_counts(map_results)
        
        return final_counts
    
    def word_count_sequential(self, documents: List[str]) -> Counter:
        """
        Sequential word count for comparison
        """
        all_words = []
        for doc in documents:
            words = self._extract_words(doc)
            all_words.extend(words)
        
        return Counter(all_words)
    
    def keyword_analysis_parallel(self, documents: List[str], keywords: List[str], 
                                 chunk_size: int = 100) -> Dict[str, int]:
        """
        Parallel keyword frequency analysis
        """
        chunks = self._chunk_documents(documents, chunk_size)
        
        # Create keyword analysis tasks
        tasks = [(chunk, keywords) for chunk in chunks]
        
        try:
            with mp.Pool(self.num_workers) as pool:
                map_results = pool.starmap(self._map_keyword_analysis, tasks)
        except Exception:
            # Fallback to sequential processing if parallel fails
            map_results = [self._map_keyword_analysis(chunk, keywords) for chunk, keywords in tasks]
        
        # Reduce results
        final_results = defaultdict(int)
        for result in map_results:
            for keyword, count in result.items():
                final_results[keyword] += count
        
        return dict(final_results)
    
    def _chunk_documents(self, documents: List[str], chunk_size: int) -> List[List[str]]:
        """Split documents into chunks for parallel processing"""
        if not documents:
            return []
        
        chunk_size = max(1, chunk_size)  # Ensure chunk_size is at least 1
        chunks = []
        for i in range(0, len(documents), chunk_size):
            chunks.append(documents[i:i + chunk_size])
        return chunks
    
    def _map_word_count(self, chunk: List[str]) -> Counter:
        """Map function: count words in a chunk"""
        word_count = Counter()
        for document in chunk:
            words = self._extract_words(document)
            word_count.update(words)
        return word_count
    
    def _map_keyword_analysis(self, chunk: List[str], keywords: List[str]) -> Dict[str, int]:
        """Map function: count keywords in a chunk"""
        keyword_counts = {keyword: 0 for keyword in keywords}
        
        for document in chunk:
            doc_lower = document.lower()
            for keyword in keywords:
                keyword_counts[keyword] += doc_lower.count(keyword.lower())
        
        return keyword_counts
    
    def _reduce_word_counts(self, map_results: List[Counter]) -> Counter:
        """Reduce function: combine word counts"""
        return sum(map_results, Counter())
    
    def _extract_words(self, text: str, use_nltk: bool = True) -> List[str]:
        """
        Extract words from text with advanced NLTK preprocessing
        Implemented by: Anurag
        """
        if use_nltk:
            try:
                # Use NLTK tokenization
                tokens = word_tokenize(text.lower())
                
                # Get English stopwords
                stop_words = set(stopwords.words('english'))
                
                # Filter tokens: remove stopwords, punctuation, short words
                words = [
                    self.stemmer.stem(word) for word in tokens 
                    if word.isalpha() and len(word) > 2 and word not in stop_words
                ]
                return words
            except Exception:
                # Fallback to basic processing if NLTK fails
                pass
        
        # Basic preprocessing as fallback
        text = re.sub(r'[^\w\s]', '', text.lower())
        words = [word for word in text.split() if word and len(word) > 2]
        return words
    
    def parallel_text_processing(self, documents: List[str], processing_func, 
                                chunk_size: int = 100, **kwargs) -> Any:
        """
        Generic parallel text processing framework
        """
        chunks = self._chunk_documents(documents, chunk_size)
        
        # Prepare tasks with additional arguments
        if kwargs:
            tasks = [(chunk, kwargs) for chunk in chunks]
            with mp.Pool(self.num_workers) as pool:
                results = pool.starmap(processing_func, tasks)
        else:
            with mp.Pool(self.num_workers) as pool:
                results = pool.map(processing_func, chunks)
        
        return results
    
    def benchmark_performance(self, documents: List[str], operation: str = "word_count") -> Dict[str, Any]:
        """
        Benchmark parallel vs sequential performance
        """
        results = {}
        
        if operation == "word_count":
            # Sequential timing
            start_time = time.time()
            seq_result = self.word_count_sequential(documents)
            results['sequential_time'] = time.time() - start_time
            
            # Parallel timing
            start_time = time.time()
            par_result = self.word_count_parallel(documents)
            results['parallel_time'] = time.time() - start_time
            
            # Verify results are equivalent
            results['results_match'] = seq_result == par_result
        
        # Calculate speedup with error handling
        if 'sequential_time' in results and 'parallel_time' in results and results['parallel_time'] > 0:
            results['speedup'] = results['sequential_time'] / results['parallel_time']
            results['efficiency'] = results['speedup'] / self.num_workers
        else:
            results['speedup'] = 1.0
            results['efficiency'] = 1.0
        
        return results
    
    def distributed_processing_simulation(self, documents: List[str], 
                                        num_nodes: int = 3) -> Dict[str, Any]:
        """
        Simulate distributed processing across multiple nodes
        """
        # Simulate node distribution
        node_chunks = self._distribute_to_nodes(documents, num_nodes)
        
        start_time = time.time()
        
        # Process each "node" in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_nodes) as executor:
            node_futures = []
            for node_id, node_data in enumerate(node_chunks):
                future = executor.submit(self._process_node, node_id, node_data)
                node_futures.append(future)
            
            # Collect results from all nodes
            node_results = []
            for future in concurrent.futures.as_completed(node_futures):
                node_results.append(future.result())
        
        processing_time = time.time() - start_time
        
        # Aggregate results from all nodes
        final_result = self._aggregate_node_results(node_results)
        
        return {
            'processing_time': processing_time,
            'num_nodes': num_nodes,
            'node_results': node_results,
            'final_result': final_result,
            'documents_per_node': [len(chunk) for chunk in node_chunks]
        }
    
    def _distribute_to_nodes(self, documents: List[str], num_nodes: int) -> List[List[str]]:
        """Distribute documents across simulated nodes"""
        chunk_size = len(documents) // num_nodes
        chunks = []
        
        for i in range(num_nodes):
            start_idx = i * chunk_size
            if i == num_nodes - 1:  # Last node gets remaining documents
                end_idx = len(documents)
            else:
                end_idx = (i + 1) * chunk_size
            
            chunks.append(documents[start_idx:end_idx])
        
        return chunks
    
    def _process_node(self, node_id: int, documents: List[str]) -> Dict[str, Any]:
        """Process documents on a simulated node"""
        start_time = time.time()
        
        # Perform word count on this node
        word_counts = self.word_count_sequential(documents)
        
        processing_time = time.time() - start_time
        
        return {
            'node_id': node_id,
            'processing_time': processing_time,
            'document_count': len(documents),
            'word_counts': word_counts,
            'total_words': sum(word_counts.values())
        }
    
    def _aggregate_node_results(self, node_results: List[Dict]) -> Counter:
        """Aggregate results from all nodes"""
        final_counts = Counter()
        
        for node_result in node_results:
            final_counts.update(node_result['word_counts'])
        
        return final_counts
