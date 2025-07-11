module.exports = {
  apps: [
    {
      name: 'streamlit-text-processor',
      script: 'streamlit',
      args: 'run app.py --server.port=5000 --server.address=0.0.0.0 --server.headless=true --server.enableCORS=false --server.enableXsrfProtection=false',
      cwd: './',
      interpreter: 'python',
      interpreter_args: '-u',
      instances: 1,
      autorestart: true,
      watch: false,
      max_memory_restart: '2G',
      min_uptime: '10s',
      max_restarts: 10,
      restart_delay: 4000,
      env: {
        NODE_ENV: 'production',
        PYTHONPATH: '.',
        PYTHONUNBUFFERED: '1',
        STREAMLIT_SERVER_PORT: '5000',
        STREAMLIT_SERVER_ADDRESS: '0.0.0.0',
        STREAMLIT_SERVER_HEADLESS: 'true',
        STREAMLIT_SERVER_ENABLE_CORS: 'false',
        STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION: 'false',
        STREAMLIT_BROWSER_GATHER_USAGE_STATS: 'false',
        STREAMLIT_GLOBAL_DEVELOPMENT_MODE: 'false'
      },
      env_production: {
        NODE_ENV: 'production',
        PYTHONPATH: '.',
        PYTHONUNBUFFERED: '1',
        STREAMLIT_SERVER_PORT: '5000',
        STREAMLIT_SERVER_ADDRESS: '0.0.0.0',
        STREAMLIT_SERVER_HEADLESS: 'true',
        STREAMLIT_SERVER_ENABLE_CORS: 'false',
        STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION: 'false',
        STREAMLIT_BROWSER_GATHER_USAGE_STATS: 'false',
        STREAMLIT_GLOBAL_DEVELOPMENT_MODE: 'false'
      },
      log_date_format: 'YYYY-MM-DD HH:mm:ss Z',
      error_file: './logs/streamlit-error.log',
      out_file: './logs/streamlit-out.log',
      log_file: './logs/streamlit-combined.log',
      pid_file: './logs/streamlit.pid',
      merge_logs: true,
      log_type: 'json',
      rotate_logs: true,
      max_log_size: '10M',
      retain_logs: 5,
      health_check_http: {
        url: 'http://localhost:5000/_stcore/health',
        interval: 30000,
        timeout: 5000,
        max_retries: 3,
        retry_delay: 5000
      },
      kill_timeout: 5000,
      listen_timeout: 8000,
      wait_ready: true,
      ready_timeout: 10000,
      shutdown_with_message: true,
      disable_source_map_support: true,
      source_map_support: false,
      instance_var: 'INSTANCE_ID',
      exec_mode: 'fork',
      combine_logs: true,
      automation: false,
      vizion: false,
      post_update: ['echo "Application updated successfully"'],
      pre_launch_delay: 2000,
      cron_restart: '0 2 * * *', // Restart daily at 2 AM
      monitoring: {
        http: true,
        https: false,
        port: 5000,
        path: '/_stcore/health'
      },
      exp_backoff_restart_delay: 100,
      force: true
    }
  ],
  
  deploy: {
    production: {
      user: 'node',
      host: 'localhost',
      ref: 'origin/main',
      repo: 'git@github.com:username/streamlit-text-processor.git',
      path: '/var/www/streamlit-text-processor',
      'pre-deploy-local': '',
      'post-deploy': 'npm install && pm2 reload ecosystem.config.js --env production',
      'pre-setup': '',
      'ssh_options': 'StrictHostKeyChecking=no'
    }
  }
};
