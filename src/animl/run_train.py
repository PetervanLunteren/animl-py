import sys
import subprocess
import os

if __name__ == "__main__":
    # Build command to run your usual command with args passed through
    cmd = [
        sys.executable,
        "-m",
        "animl.train",
        "--config=C:\\Peter\\training-utils\\scripts\\config.yaml",
    ] + sys.argv[1:]
    
    # Optionally, set environment variable (like PYTHONWARNINGS)
    env = os.environ.copy()
    env["PYTHONWARNINGS"] = "ignore::RuntimeWarning"
    
    # Run the command, inheriting your environment vars + args
    subprocess.run(cmd, check=True, env=env)