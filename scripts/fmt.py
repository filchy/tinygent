import subprocess
import sys


def main():
    cmds = [['uv', 'run', 'black', '.'], ['uv', 'run', 'ruff', 'format', '.']]

    exit_code = 0
    for cmd in cmds:
        result = subprocess.run(cmd)
        if result.returncode != 0:
            exit_code = result.returncode

    sys.exit(exit_code)
