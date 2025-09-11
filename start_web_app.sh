#!/bin/bash

echo "🚀 Sonar Rock vs Mine Detection Web Application"
echo "=============================================="
echo ""
echo "Starting the web server..."
echo "🌐 Web interface will be available at: http://localhost:8000"
echo "🛑 Press Ctrl+C to stop the server"
echo ""

# Check if the data file exists
if [ ! -f "sonar data.csv" ]; then
    echo "⚠️  Warning: sonar data.csv not found. The app will use dummy data for demonstration."
    echo ""
fi

# Start the application
python3 simple_app.py