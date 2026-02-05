import sys
print(f"Python Executable: {sys.executable}")
try:
    import flwr
    print(f"Flower version: {flwr.__version__}")
    import flwr.simulation
    print("flwr.simulation imported successfully")
except ImportError as e:
    print(f"ImportError: {e}")
except Exception as e:
    print(f"Error: {e}")
