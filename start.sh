#!/bin/bash

# Streamlit Text Processor PM2 Startup Script
# This script initializes and starts the Streamlit application using PM2

set -e

echo "🚀 Starting Streamlit Text Processor with PM2..."

# Check if PM2 is installed
if ! command -v pm2 &> /dev/null; then
    echo "❌ PM2 is not installed. Please install PM2 first:"
    echo "npm install -g pm2"
    exit 1
fi

# Check if Python is available
if ! command -v python &> /dev/null && ! command -v python3 &> /dev/null; then
    echo "❌ Python is not installed. Please install Python 3.8+ first."
    exit 1
fi

# Check if Streamlit is available
if ! python -c "import streamlit" 2>/dev/null && ! python3 -c "import streamlit" 2>/dev/null; then
    echo "❌ Streamlit is not installed. Please install Streamlit first:"
    echo "pip install streamlit"
    exit 1
fi

# Create logs directory if it doesn't exist
mkdir -p logs

# Load environment variables if .env.production exists
if [ -f ".env.production" ]; then
    echo "📋 Loading production environment variables..."
    export $(cat .env.production | grep -v '^#' | xargs)
fi

# Stop existing processes if running
echo "🛑 Stopping existing processes..."
pm2 stop streamlit-text-processor 2>/dev/null || true
pm2 delete streamlit-text-processor 2>/dev/null || true

# Start the application
echo "🔄 Starting Streamlit Text Processor..."
pm2 start ecosystem.config.js --env production

# Save PM2 configuration
echo "💾 Saving PM2 configuration..."
pm2 save

# Setup PM2 startup script
echo "⚙️ Setting up PM2 startup script..."
pm2 startup | tail -1 | bash 2>/dev/null || true

# Show status
echo "📊 Application status:"
pm2 status

echo "✅ Streamlit Text Processor started successfully!"
echo "🌐 Application is running at: http://localhost:5000"
echo "📈 Monitor with: pm2 monit"
echo "📋 View logs with: pm2 logs streamlit-text-processor"
echo "🔄 Restart with: pm2 restart streamlit-text-processor"
echo "🛑 Stop with: pm2 stop streamlit-text-processor"

# Run health check
echo "🏥 Running health check..."
sleep 5
python health-check.py
