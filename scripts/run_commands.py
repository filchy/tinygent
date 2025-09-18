import subprocess


def run_commands(cmds: list[list[str]]) -> int:
    exit_code = 0
    for cmd in cmds:
        print(f'Running command: {" ".join(cmd)}')
        result = subprocess.run(cmd)
        if result.returncode != 0:
            exit_code = result.returncode

    return exit_code
