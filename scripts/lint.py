import subprocess
import sys

def main():
    cmds = [
        ['uv', 'run', 'ruff', 'check'],
        ['uv', 'run', 'mypy']
    ]

    exit_code = 0
    for cmd in cmds:
        result = subprocess.run(cmd)
        if result.returncode != 0:
            exit_code = result.returncode

    sys.exit(exit_code)
