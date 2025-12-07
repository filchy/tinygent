import importlib.metadata
import logging
import pkgutil
import time

import click
import typer

logger = logging.getLogger(__name__)


def _path_to_import_path(path: str) -> str:
    """Convert 'path-like' path to 'import-like' path."""
    from pathlib import Path
    import sys

    p = Path(path).resolve()

    if p.is_dir():
        p = p / '__init__.py'

    if p.suffix != '.py':
        raise ValueError(f'Path is not a Python module: {path}')

    for root in map(Path, sys.path):
        try:
            rel = p.relative_to(root.resolve())
            rel = rel.with_suffix('')
            return '.'.join(rel.parts)
        except ValueError:
            continue

    raise ValueError(f'Path {path} is not under any sys.path entry')


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


def discover_entry_points(
    groups: str | list[str],
) -> list[importlib.metadata.EntryPoint]:
    """Discover entry points for 'tinygent'."""
    entry_points = importlib.metadata.entry_points()

    if isinstance(groups, str):
        return list(entry_points.select(group=groups))

    discovered: list[importlib.metadata.EntryPoint] = []
    for group in groups:
        discovered.extend(entry_points.select(group=group))
    return discovered


def create_entry_point_from_path(path: str) -> importlib.metadata.EntryPoint:
    """Create new entry point from 'folder-like' path."""
    import_path = _path_to_import_path(path)

    return importlib.metadata.EntryPoint(
        name=import_path.replace('.', '_'), value=import_path, group='manual'
    )


def discover_and_register_components(additional_paths: list[str] | str = []) -> None:
    """Discover and register components from the 'tinygent' package."""
    entry_points = discover_entry_points(['components', 'functions'])
    entry_points.extend(
        [
            create_entry_point_from_path(p)
            for p in (
                additional_paths
                if isinstance(additional_paths, list)
                else [additional_paths]
            )
        ]
    )

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
