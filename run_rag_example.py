#!/usr/bin/env python3
"""
Launcher script for the RAG example.

Run this from the project root to avoid import issues.
"""

import sys
import os

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import and run the example
from ragstore.example_usage import main

if __name__ == "__main__":
    main() 