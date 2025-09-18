import sys

from .run_commands import run_commands


def main():
    cmds = [['uv', 'run', 'ruff', 'check'], ['uv', 'run', 'mypy']]

    sys.exit(run_commands(cmds))
