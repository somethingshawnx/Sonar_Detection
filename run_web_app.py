#!/usr/bin/env python3
"""
Sonar Rock vs Mine Detection Web Application
Run this script to start the web server
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False
    return True

def start_app():
    """Start the Flask application"""
    print("Starting Sonar Detection Web Application...")
    print("🌐 Web interface will be available at: http://localhost:5000")
    print("📊 Model will be trained automatically on startup")
    print("🛑 Press Ctrl+C to stop the server")
    print("-" * 50)
    
    try:
        # Start the Flask app
        os.system("python app.py")
    except KeyboardInterrupt:
        print("\n👋 Server stopped. Goodbye!")
    except Exception as e:
        print(f"❌ Error starting server: {e}")

if __name__ == "__main__":
    print("🚀 Sonar Rock vs Mine Detection Web App")
    print("=" * 50)
    
    # Check if requirements.txt exists
    if not os.path.exists("requirements.txt"):
        print("❌ requirements.txt not found!")
        sys.exit(1)
    
    # Install requirements
    if install_requirements():
        start_app()
    else:
        print("❌ Failed to install requirements. Please install manually:")
        print("pip install -r requirements.txt")
        sys.exit(1)