from pathlib import Path
import subprocess

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class FrontendBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        frontend_dir = Path(__file__).parent / 'src' / 'tiny_chat' / 'frontend'

        if not frontend_dir.is_dir():
            return

        subprocess.check_call(['npm', 'ci'], cwd=frontend_dir)
        subprocess.check_call(['npm', 'run', 'build'], cwd=frontend_dir)
