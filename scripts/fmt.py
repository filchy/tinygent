import sys

from tinygent.utils.run_commands import run_commands


def main():
    cmds = [['uv', 'run', 'black', '.'], ['uv', 'run', 'ruff', 'format', '.']]

    sys.exit(run_commands(cmds))
