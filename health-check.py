"""
Health Check Script for Streamlit Text Processor
This script performs comprehensive health checks on the running application
"""

import requests
import sys
import time
import os
import json
from datetime import datetime
import subprocess

class HealthChecker:
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url
        self.timeout = 10
        self.max_retries = 3
        
    def check_http_endpoint(self):
        """Check if the main HTTP endpoint is responding"""
        print("🌐 Checking HTTP endpoint...")
        
        for attempt in range(self.max_retries):
            try:
                response = requests.get(
                    f"{self.base_url}/_stcore/health",
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    print(f"✅ HTTP endpoint is healthy (Status: {response.status_code})")
                    return True
                else:
                    print(f"⚠️ HTTP endpoint returned status: {response.status_code}")
                    
            except requests.exceptions.ConnectionError:
                print(f"❌ Connection failed (Attempt {attempt + 1}/{self.max_retries})")
            except requests.exceptions.Timeout:
                print(f"⏱️ Request timeout (Attempt {attempt + 1}/{self.max_retries})")
            except Exception as e:
                print(f"❌ Unexpected error: {str(e)} (Attempt {attempt + 1}/{self.max_retries})")
            
            if attempt < self.max_retries - 1:
                time.sleep(2)
        
        return False
    
    def check_streamlit_app(self):
        """Check if the Streamlit app page loads"""
        print("📱 Checking Streamlit application...")
        
        try:
            response = requests.get(
                self.base_url,
                timeout=self.timeout,
                allow_redirects=True
            )
            
            if response.status_code == 200:
                # Check if it contains Streamlit-specific content
                content = response.text.lower()
                if 'streamlit' in content or 'st-' in content:
                    print("✅ Streamlit application is responding correctly")
                    return True
                else:
                    print("⚠️ Response received but doesn't appear to be Streamlit")
                    return False
            else:
                print(f"❌ Streamlit app returned status: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Failed to check Streamlit app: {str(e)}")
            return False
    
    def check_pm2_process(self):
        """Check PM2 process status"""
        print("🔧 Checking PM2 process status...")
        
        try:
            # Get PM2 process list
            result = subprocess.run(
                ['pm2', 'jlist'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                processes = json.loads(result.stdout)
                
                # Find our process
                target_process = None
                for process in processes:
                    if process.get('name') == 'streamlit-text-processor':
                        target_process = process
                        break
                
                if target_process:
                    status = target_process.get('pm2_env', {}).get('status', 'unknown')
                    cpu = target_process.get('monit', {}).get('cpu', 0)
                    memory = target_process.get('monit', {}).get('memory', 0)
                    uptime = target_process.get('pm2_env', {}).get('pm_uptime', 0)
                    
                    memory_mb = memory / (1024 * 1024) if memory else 0
                    uptime_seconds = (time.time() * 1000 - uptime) / 1000 if uptime else 0
                    
                    print(f"📊 Process Status: {status}")
                    print(f"🧠 CPU Usage: {cpu}%")
                    print(f"💾 Memory Usage: {memory_mb:.1f} MB")
                    print(f"⏰ Uptime: {uptime_seconds:.0f} seconds")
                    
                    if status == 'online':
                        print("✅ PM2 process is running correctly")
                        return True
                    else:
                        print(f"❌ PM2 process status is not online: {status}")
                        return False
                else:
                    print("❌ streamlit-text-processor process not found in PM2")
                    return False
            else:
                print(f"❌ Failed to get PM2 process list: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print("⏱️ PM2 command timed out")
            return False
        except FileNotFoundError:
            print("❌ PM2 not found. Is PM2 installed?")
            return False
        except Exception as e:
            print(f"❌ Error checking PM2 process: {str(e)}")
            return False
    
    def check_log_files(self):
        """Check if log files are being written to"""
        print("📋 Checking log files...")
        
        log_files = [
            'logs/streamlit-out.log',
            'logs/streamlit-error.log',
            'logs/streamlit-combined.log'
        ]
        
        healthy = True
        
        for log_file in log_files:
            if os.path.exists(log_file):
                stat = os.stat(log_file)
                size = stat.st_size
                modified = stat.st_mtime
                
                # Check if file was modified in the last 5 minutes
                time_diff = time.time() - modified
                
                if time_diff < 300:  # 5 minutes
                    print(f"✅ {log_file}: {size} bytes (recently updated)")
                else:
                    print(f"⚠️ {log_file}: {size} bytes (last updated {time_diff:.0f}s ago)")
            else:
                print(f"❌ {log_file}: File not found")
                healthy = False
        
        return healthy
    
    def check_dependencies(self):
        """Check if required Python packages are available"""
        print("📦 Checking Python dependencies...")
        
        required_packages = [
            'streamlit',
            'pandas',
            'plotly',
            'psutil',
            'nltk'
        ]
        
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package)
                print(f"✅ {package} is available")
            except ImportError:
                print(f"❌ {package} is missing")
                missing_packages.append(package)
        
        if missing_packages:
            print(f"❌ Missing packages: {', '.join(missing_packages)}")
            return False
        
        print("✅ All required dependencies are available")
        return True
    
    def run_comprehensive_check(self):
        """Run all health checks"""
        print("🏥 Starting comprehensive health check...")
        print(f"🕐 Timestamp: {datetime.now().isoformat()}")
        print("=" * 50)
        
        checks = [
            ("Dependencies", self.check_dependencies),
            ("PM2 Process", self.check_pm2_process),
            ("HTTP Endpoint", self.check_http_endpoint),
            ("Streamlit App", self.check_streamlit_app),
            ("Log Files", self.check_log_files)
        ]
        
        passed = 0
        total = len(checks)
        
        for check_name, check_func in checks:
            print(f"\n🔍 Running {check_name} check...")
            try:
                if check_func():
                    passed += 1
                    print(f"✅ {check_name} check passed")
                else:
                    print(f"❌ {check_name} check failed")
            except Exception as e:
                print(f"❌ {check_name} check failed with error: {str(e)}")
        
        print("\n" + "=" * 50)
        print(f"🏁 Health check completed: {passed}/{total} checks passed")
        
        if passed == total:
            print("🎉 All health checks passed! Application is healthy.")
            return True
        else:
            print("⚠️ Some health checks failed. Please investigate.")
            return False

def main():
    """Main health check function"""
    checker = HealthChecker()
    
    # Check if we can reach the application
    if checker.run_comprehensive_check():
        print("\n✅ Health check successful!")
        sys.exit(0)
    else:
        print("\n❌ Health check failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
