def run_cli() -> None:
    import os
    import sys

    parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)

    from tinygent.cli.entry_point import app

    app()


if __name__ == '__main__':
    run_cli()
