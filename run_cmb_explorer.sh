#!/bin/bash
# Quick start script for CMB Power Spectrum Explorer

echo "=========================================="
echo "Interactive CMB Power Spectrum Explorer"
echo "=========================================="
echo ""

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null
then
    echo "Streamlit not found. Installing requirements..."
    pip install -r requirements.txt
fi

echo "Starting the application..."
echo ""
echo "The app will open in your default browser."
echo "If it doesn't, navigate to: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the application."
echo ""

streamlit run cmb_explorer_app.py
