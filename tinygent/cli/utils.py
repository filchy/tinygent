import importlib
import pkgutil

import click
import typer


def get_click_context() -> click.Context:
    """Get the current Click context."""
    ctx = click.get_current_context(silent=True)
    if ctx is None:
        raise RuntimeError('No Click context found')
    return ctx


def register_commands_from_package(app: typer.Typer, package: str):
    """Dynamically register commands from a package to a Typer app."""
    package_module = importlib.import_module(package)

    for _, module_name, is_pkg in pkgutil.iter_modules(package_module.__path__):
        if is_pkg:
            continue

        module = importlib.import_module(f'{package}.{module_name}')

        if hasattr(module, 'main') and callable(module.main):
            app.command(name=module_name)(module.main)
