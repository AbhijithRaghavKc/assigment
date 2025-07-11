# Parallel Text Processing System

A comprehensive Python-based parallel text processing system built with Streamlit that demonstrates various parallel computing concepts including MapReduce, stream processing, and sentiment analysis. The application provides an interactive web interface for benchmarking different parallel processing approaches and visualizing performance metrics in real-time.

## Features

- **MapReduce Processing**: Parallel word counting and keyword analysis using the MapReduce pattern
- **Stream Processing**: Real-time text analysis with sliding window operations
- **Sentiment Analysis**: Rule-based sentiment analysis with parallel execution capabilities
- **Performance Monitoring**: Real-time CPU, memory, throughput, and latency metrics
- **Interactive Visualizations**: Plotly charts for performance analysis and results display
- **Multi-format Data Support**: Text files, CSV, and JSON data ingestion

## System Architecture

### Frontend
- **Framework**: Streamlit web application
- **Interface**: Multi-tab layout with interactive controls
- **Visualization**: Plotly charts for real-time performance monitoring

### Backend
- **Processing Engine**: Python multiprocessing and concurrent.futures
- **Design Pattern**: Modular processor classes with clear separation of concerns
- **Parallelization**: Multiple strategies including MapReduce, stream processing, and parallel sentiment analysis

## Quick Start

### Local Development

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd parallel-text-processor
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app.py --server.port 5000
   ```

4. **Access the application**
   Open your browser to `http://localhost:5000`

## EC2 Deployment

### Prerequisites
- EC2 instance with Amazon Linux 2 or Ubuntu
- Security group allowing inbound traffic on port 5000
- SSH access to your EC2 instance

### Automated Setup

1. **Upload files to EC2**
   ```bash
   scp -i your-key.pem -r . ec2-user@your-ec2-ip:/home/ec2-user/parallel-text-processor/
   ```

2. **Run setup script**
   ```bash
   ssh -i your-key.pem ec2-user@your-ec2-ip
   cd parallel-text-processor
   chmod +x setup-ec2.sh
   bash setup-ec2.sh
   ```

3. **Start the application with PM2**
   ```bash
   pm2 start ecosystem.config.js
   pm2 save
   ```

4. **Enable auto-startup on boot**
   ```bash
   sudo systemctl enable pm2-ec2-user
   ```

### Manual Setup

1. **Install system dependencies**
   ```bash
   sudo yum update -y
   sudo yum install -y python3.11 python3.11-pip
   curl -fsSL https://rpm.nodesource.com/setup_18.x | sudo bash -
   sudo yum install -y nodejs
   sudo npm install -g pm2
   ```

2. **Install Python dependencies**
   ```bash
   python3.11 -m pip install -r requirements.txt
   ```

3. **Download NLTK data**
   ```bash
   python3.11 -c "
   import nltk
   nltk.download('punkt', quiet=True)
   nltk.download('stopwords', quiet=True)
   nltk.download('wordnet', quiet=True)
   "
   ```

4. **Start with PM2**
   ```bash
   pm2 start ecosystem.config.js
   ```

## PM2 Management Commands

```bash
# Start the application
pm2 start ecosystem.config.js

# Stop the application
pm2 stop parallel-text-processor

# Restart the application
pm2 restart parallel-text-processor

# View logs
pm2 logs parallel-text-processor

# Monitor processes
pm2 monit

# Save current PM2 configuration
pm2 save

# View process status
pm2 status
```

## Application Usage

### 1. Data Ingestion Tab
- Upload text files, CSV, or JSON data
- Use sample datasets for demonstration
- Configure processing parameters (workers, batch size, chunk size)

### 2. MapReduce Tab
- Perform parallel word counting
- Analyze keyword frequencies
- Compare parallel vs sequential performance
- View processing metrics and visualizations

### 3. Stream Processing Tab
- Simulate real-time text processing
- Configure sliding window operations
- Monitor word frequency trends
- Analyze sentiment in real-time

### 4. Performance Analysis Tab
- Compare different processing methods
- Analyze scalability metrics
- View throughput and latency statistics
- Optimize worker configurations

### 5. Dashboard Tab
- Real-time performance monitoring
- System resource utilization
- Processing history and trends
- Export performance metrics

## Configuration

### Environment Variables
- `STREAMLIT_SERVER_PORT`: Application port (default: 5000)
- `STREAMLIT_SERVER_ADDRESS`: Bind address (default: 0.0.0.0)
- `STREAMLIT_BROWSER_GATHER_USAGE_STATS`: Disable usage stats (false)

### PM2 Configuration
Edit `ecosystem.config.js` to modify:
- Memory limits
- Instance count
- Environment variables
- Log file locations

## Performance Tuning

### For Small Datasets (< 1MB)
- Use 2-4 workers
- Batch size: 50-100
- Chunk size: 50-100

### For Medium Datasets (1-10MB)
- Use 4-8 workers
- Batch size: 100-500
- Chunk size: 100-200

### For Large Datasets (> 10MB)
- Use 8+ workers (up to CPU count)
- Batch size: 500-1000
- Chunk size: 200-500

## Troubleshooting

### Common Issues

1. **Port already in use**
   ```bash
   sudo lsof -i :5000
   sudo kill -9 <PID>
   ```

2. **Permission denied errors**
   ```bash
   sudo chown -R ec2-user:ec2-user /home/ec2-user/parallel-text-processor
   ```

3. **NLTK data missing**
   ```bash
   python3.11 -c "import nltk; nltk.download('all')"
   ```

4. **PM2 not starting on boot**
   ```bash
   pm2 startup
   pm2 save
   ```

### Log Locations
- PM2 logs: `./logs/`
- Application logs: Check PM2 logs with `pm2 logs`
- System logs: `/var/log/messages`

## Security Considerations

1. **Firewall Configuration**
   ```bash
   sudo ufw allow 5000
   sudo ufw enable
   ```

2. **Reverse Proxy Setup** (Recommended for production)
   Configure Nginx or Apache as a reverse proxy

3. **SSL/TLS Certificate**
   Use Let's Encrypt or AWS Certificate Manager

## Team Attribution

### Anurag's Contributions
- MapReduce processing implementation
- Sentiment analysis with parallel processing
- Performance monitoring and benchmarking
- MapReduce and Performance Analysis tabs

### Abhijith's Contributions
- Stream processing with sliding windows
- Data loading and ingestion utilities
- Visualization and charts
- Data Ingestion, Stream Processing, and Dashboard tabs

## Dependencies

### Core Libraries
- `streamlit==1.28.1`: Web application framework
- `pandas==2.1.4`: Data manipulation and analysis
- `plotly==5.17.0`: Interactive visualization
- `psutil==5.9.6`: System performance monitoring
- `nltk==3.8.1`: Natural language processing

### System Requirements
- Python 3.11+
- Node.js 18+ (for PM2)
- 2GB+ RAM (4GB recommended)
- 2+ CPU cores (4+ recommended)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review PM2 logs: `pm2 logs parallel-text-processor`
3. Check system resources: `htop` or `pm2 monit`
4. Verify network connectivity and security groups

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes and test thoroughly
4. Submit a pull request with detailed description

## Changelog

- **June 28, 2025**: Initial release with comprehensive parallel processing features
- **June 28, 2025**: Added EC2 deployment configuration and PM2 setup
- **June 28, 2025**: Enhanced error handling and system stability