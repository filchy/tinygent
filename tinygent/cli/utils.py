import importlib.metadata
import logging
import pkgutil
import time

import click
import typer

logger = logging.getLogger(__name__)


def get_click_context() -> click.Context:
    """Get the current Click context."""
    ctx = click.get_current_context(silent=True)
    if ctx is None:
        raise RuntimeError('No Click context found')
    return ctx


def register_commands_from_package(app: typer.Typer, package: str) -> None:
    """Dynamically register commands from a package to a Typer app."""
    package_module = importlib.import_module(package)

    for _, module_name, is_pkg in pkgutil.iter_modules(package_module.__path__):
        if is_pkg:
            continue

        module = importlib.import_module(f'{package}.{module_name}')

        if hasattr(module, 'main') and callable(module.main):
            help_text = module.__doc__ or f'{module_name} command'
            app.command(name=module_name, help=help_text)(module.main)


def discover_entry_points(group: str) -> list[importlib.metadata.EntryPoint]:
    """Discover entry points for 'tinygent'."""
    entry_points = importlib.metadata.entry_points()

    return list(entry_points.select(group=group))


def discover_and_register_components() -> None:
    """Discover and register components from the 'tinygent' package."""
    entry_points = discover_entry_points('components')

    count = 0
    for entry_point in entry_points:
        try:
            logger.debug('Loading component %d: %s', count + 1, entry_point.name)

            start_time = time.time()

            entry_point.load()

            logger.debug(
                'Loading module %s from entry point %s ... Complete (%.2f s)',
                entry_point.module,
                entry_point.name,
                time.time() - start_time,
            )
        except ImportError:
            logger.warning('Failed to import plugin %s', entry_point.name, exc_info=True)
        except Exception as e:
            logger.error(
                'Error loading plugin %s: %s', entry_point.name, str(e), exc_info=True
            )
        finally:
            count += 1
