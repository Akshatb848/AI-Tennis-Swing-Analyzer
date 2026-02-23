import subprocess
import sys

if __name__ == "__main__":
    subprocess.run([sys.executable, "-m", "uvicorn", "services.api:app", "--host", "0.0.0.0", "--port", "8000"])
