# ============================================================================
# Performance Monitor - Implemented by: Anurag
# 
# This module provides comprehensive performance monitoring and benchmarking
# capabilities. It tracks CPU, memory usage, throughput, and latency metrics
# for parallel processing operations.
# ============================================================================

import time
import psutil
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from collections import defaultdict, deque
import multiprocessing as mp

class PerformanceMonitor:
    """
    Performance monitoring and benchmarking utilities
    Tracks CPU, memory, throughput, and latency metrics
    """
    
    def __init__(self):
        self.metrics_history = defaultdict(deque)
        self.monitoring_active = False
        self.monitor_thread = None
        self.start_time = None
        
    def start_monitoring(self, interval: float = 1.0):
        """
        Start continuous performance monitoring
        """
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.start_time = time.time()
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """
        Stop performance monitoring
        """
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def _monitoring_loop(self, interval: float):
        """
        Continuous monitoring loop
        """
        while self.monitoring_active:
            timestamp = datetime.now()
            
            # Collect system metrics with error handling
            try:
                cpu_percent = psutil.cpu_percent(interval=None)
                memory = psutil.virtual_memory()
            except Exception:
                cpu_percent = 0.0
                # Create a simple object with required attributes
                class MockMemory:
                    def __init__(self):
                        self.percent = 0.0
                        self.used = 0
                        self.available = 0
                memory = MockMemory()
            
            # Store metrics
            self.metrics_history['timestamp'].append(timestamp)
            self.metrics_history['cpu_percent'].append(cpu_percent)
            self.metrics_history['memory_percent'].append(memory.percent)
            self.metrics_history['memory_used_gb'].append(memory.used / (1024**3))
            self.metrics_history['memory_available_gb'].append(memory.available / (1024**3))
            
            # Keep only last 1000 data points
            max_points = 1000
            for key in self.metrics_history:
                if len(self.metrics_history[key]) > max_points:
                    self.metrics_history[key].popleft()
            
            time.sleep(interval)
    
    def benchmark_operation(self, operation_func, *args, **kwargs) -> Dict[str, Any]:
        """
        Benchmark a specific operation with detailed metrics
        """
        # Pre-operation metrics
        pre_cpu = psutil.cpu_percent(interval=None)
        pre_memory = psutil.virtual_memory()
        
        # Start timing
        start_time = time.time()
        start_process_time = time.process_time()
        
        try:
            # Execute operation
            result = operation_func(*args, **kwargs)
            success = True
            error_message = None
        except Exception as e:
            result = None
            success = False
            error_message = str(e)
        
        # End timing
        end_time = time.time()
        end_process_time = time.process_time()
        
        # Post-operation metrics
        post_cpu = psutil.cpu_percent(interval=None)
        post_memory = psutil.virtual_memory()
        
        # Calculate metrics
        wall_time = end_time - start_time
        cpu_time = end_process_time - start_process_time
        memory_delta = post_memory.used - pre_memory.used
        
        return {
            'success': success,
            'result': result,
            'error_message': error_message,
            'wall_time': wall_time,
            'cpu_time': cpu_time,
            'cpu_utilization': (post_cpu + pre_cpu) / 2,
            'memory_delta_mb': memory_delta / (1024**2),
            'peak_memory_mb': post_memory.used / (1024**2),
            'timestamp': datetime.now()
        }
    
    def measure_throughput(self, operation_func, data_items: List[Any], 
                          batch_sizes: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Measure throughput for different batch sizes
        """
        if batch_sizes is None:
            batch_sizes = [1, 10, 50, 100, 500, 1000]
        
        throughput_results = {}
        
        for batch_size in batch_sizes:
            if batch_size > len(data_items):
                continue
            
            # Take a sample of data
            sample_data = data_items[:batch_size]
            
            # Measure processing time
            start_time = time.time()
            try:
                operation_func(sample_data)
                processing_time = time.time() - start_time
                throughput = batch_size / processing_time
                success = True
            except Exception as e:
                processing_time = time.time() - start_time
                throughput = 0
                success = False
            
            throughput_results[batch_size] = {
                'batch_size': batch_size,
                'processing_time': processing_time,
                'throughput': throughput,
                'items_per_second': throughput,
                'success': success
            }
        
        # Find optimal batch size
        successful_results = {k: v for k, v in throughput_results.items() if v['success']}
        if successful_results:
            optimal_batch = max(successful_results.keys(), 
                              key=lambda k: successful_results[k]['throughput'])
        else:
            optimal_batch = None
        
        return {
            'throughput_by_batch_size': throughput_results,
            'optimal_batch_size': optimal_batch,
            'max_throughput': max([r['throughput'] for r in successful_results.values()]) if successful_results else 0
        }
    
    def measure_latency(self, operation_func, data_items: List[Any], 
                       num_samples: int = 100) -> Dict[str, Any]:
        """
        Measure latency distribution for individual operations
        """
        latencies = []
        
        # Sample random items for latency measurement
        import random
        sample_items = random.sample(data_items, min(num_samples, len(data_items)))
        
        for item in sample_items:
            start_time = time.time()
            try:
                operation_func([item])  # Process single item
                latency = time.time() - start_time
                latencies.append(latency)
            except Exception:
                continue
        
        if not latencies:
            return {'error': 'No successful latency measurements'}
        
        # Calculate statistics
        latencies.sort()
        n = len(latencies)
        
        return {
            'num_samples': n,
            'mean_latency': sum(latencies) / n,
            'median_latency': latencies[n // 2],
            'min_latency': min(latencies),
            'max_latency': max(latencies),
            'p95_latency': latencies[int(n * 0.95)] if n > 20 else max(latencies),
            'p99_latency': latencies[int(n * 0.99)] if n > 100 else max(latencies),
            'latency_distribution': latencies
        }
    
    def compare_scaling_performance(self, operation_func, data_items: List[Any], 
                                  worker_counts: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Compare performance across different numbers of workers
        """
        if worker_counts is None:
            max_workers = mp.cpu_count()
            worker_counts = [1, 2, 4, max_workers] if max_workers >= 4 else [1, max_workers]
        
        scaling_results = {}
        baseline_time = None
        
        for num_workers in worker_counts:
            start_time = time.time()
            
            try:
                # Call operation with specified number of workers
                old_workers = None
                if hasattr(operation_func, '__self__'):
                    # Method call - update num_workers if possible
                    old_workers = getattr(operation_func.__self__, 'num_workers', None)
                    if hasattr(operation_func.__self__, 'num_workers'):
                        operation_func.__self__.num_workers = num_workers
                
                result = operation_func(data_items)
                processing_time = time.time() - start_time
                
                # Restore original worker count
                if hasattr(operation_func, '__self__') and old_workers is not None:
                    operation_func.__self__.num_workers = old_workers
                
                success = True
                
            except Exception as e:
                processing_time = time.time() - start_time
                success = False
                result = None
            
            if success:
                throughput = len(data_items) / processing_time
                
                if baseline_time is None:
                    baseline_time = processing_time
                    speedup = 1.0
                else:
                    speedup = baseline_time / processing_time
                
                efficiency = speedup / num_workers
                
                scaling_results[num_workers] = {
                    'num_workers': num_workers,
                    'processing_time': processing_time,
                    'throughput': throughput,
                    'speedup': speedup,
                    'efficiency': efficiency,
                    'success': True
                }
            else:
                scaling_results[num_workers] = {
                    'num_workers': num_workers,
                    'processing_time': processing_time,
                    'success': False
                }
        
        return {
            'scaling_results': scaling_results,
            'optimal_workers': self._find_optimal_workers(scaling_results),
            'parallel_efficiency': self._calculate_parallel_efficiency(scaling_results)
        }
    
    def _find_optimal_workers(self, scaling_results: Dict) -> Optional[int]:
        """
        Find optimal number of workers based on efficiency
        """
        successful_results = {k: v for k, v in scaling_results.items() if v.get('success', False)}
        
        if not successful_results:
            return None
        
        # Find worker count with highest efficiency > 0.7 or highest throughput
        high_efficiency = {k: v for k, v in successful_results.items() 
                          if v.get('efficiency', 0) > 0.7}
        
        if high_efficiency:
            return max(high_efficiency.keys(), key=lambda k: high_efficiency[k]['throughput'])
        else:
            return max(successful_results.keys(), key=lambda k: successful_results[k]['throughput'])
    
    def _calculate_parallel_efficiency(self, scaling_results: Dict) -> float:
        """
        Calculate overall parallel efficiency
        """
        successful_results = [v for v in scaling_results.values() if v.get('success', False)]
        
        if len(successful_results) < 2:
            return 0.0
        
        efficiencies = [r['efficiency'] for r in successful_results if 'efficiency' in r]
        return sum(efficiencies) / len(efficiencies) if efficiencies else 0.0
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        Get comprehensive system information
        """
        return {
            'cpu_count': mp.cpu_count(),
            'cpu_freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'memory_available_gb': psutil.virtual_memory().available / (1024**3),
            'disk_usage': {
                'total_gb': psutil.disk_usage('/').total / (1024**3),
                'free_gb': psutil.disk_usage('/').free / (1024**3)
            },
            'python_version': "3.11",
            'platform': "Linux x86_64"
        }
    
    def export_metrics(self, format: str = 'dict') -> Any:
        """
        Export collected metrics in specified format
        """
        if format == 'dict':
            return dict(self.metrics_history)
        elif format == 'dataframe':
            import pandas as pd
            return pd.DataFrame(dict(self.metrics_history))
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def generate_performance_report(self, operation_name: str, 
                                  benchmark_results: Dict[str, Any]) -> str:
        """
        Generate a comprehensive performance report
        """
        report = f"""
# Performance Report: {operation_name}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## System Information
- CPU Cores: {mp.cpu_count()}
- Total Memory: {psutil.virtual_memory().total / (1024**3):.2f} GB
- Available Memory: {psutil.virtual_memory().available / (1024**3):.2f} GB

## Performance Metrics
"""
        
        if 'wall_time' in benchmark_results:
            report += f"- Wall Time: {benchmark_results['wall_time']:.3f} seconds\n"
            report += f"- CPU Time: {benchmark_results.get('cpu_time', 0):.3f} seconds\n"
            report += f"- CPU Utilization: {benchmark_results.get('cpu_utilization', 0):.1f}%\n"
            report += f"- Memory Delta: {benchmark_results.get('memory_delta_mb', 0):.2f} MB\n"
        
        if 'throughput_by_batch_size' in benchmark_results:
            report += "\n## Throughput Analysis\n"
            for batch_size, metrics in benchmark_results['throughput_by_batch_size'].items():
                if metrics['success']:
                    report += f"- Batch Size {batch_size}: {metrics['throughput']:.2f} items/sec\n"
        
        if 'scaling_results' in benchmark_results:
            report += "\n## Scaling Performance\n"
            for workers, metrics in benchmark_results['scaling_results'].items():
                if metrics['success']:
                    report += f"- {workers} Workers: {metrics['throughput']:.2f} items/sec (Speedup: {metrics['speedup']:.2f}x, Efficiency: {metrics['efficiency']:.2f})\n"
        
        return report
