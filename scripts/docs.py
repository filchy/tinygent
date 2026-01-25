import sys

from .run_commands import run_commands


def main():
    """Serve the MkDocs documentation locally."""
    cmds = [['uv', 'run', 'mkdocs', 'serve']]

    sys.exit(run_commands(cmds))
