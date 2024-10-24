import os
import sys
import subprocess

# Define the path to the Streamlit app
script_path = os.path.join(os.path.dirname(__file__), 'app.py')

# Run the Streamlit app
if __name__ == "__main__":
    subprocess.run([sys.executable, "-m", "streamlit", "run", script_path])
