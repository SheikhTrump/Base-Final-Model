import sys
print(f"Python: {sys.executable}")
try:
    import ray
    print(f"Ray version: {ray.__version__}")
    ray.init(ignore_reinit_error=True)
    print("Ray init success")
    ray.shutdown()
except Exception as e:
    print(f"Ray Error: {e}")

try:
    import flwr.simulation
    print("Flower Simulation import success")
except Exception as e:
    print(f"Flower Sim Error: {e}")
