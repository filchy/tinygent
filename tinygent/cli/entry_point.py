import logging
import time
from typing import Annotated
from typing import Any

import click
import typer

from tinygent.cli.utils import discover_and_register_components
from tinygent.cli.utils import get_click_context
from tinygent.cli.utils import register_commands_from_package
from tinygent.logging import LOG_LEVELS
from tinygent.logging import setup_logger

app = typer.Typer(name='tiny', help='TinyGent CLI')

register_commands_from_package(app=app, package=f'{__package__}.commands')


def cli_end(result, **kwargs: Any) -> None:
    """Callback function to be executed at the end of the CLI command."""
    logger = logging.getLogger(__name__)

    ctx = get_click_context()

    logger.info(
        'Total execution time: %.2f seconds', time.time() - ctx.obj['start_time']
    )
    logger.info('TinyGent CLI finished! Nothing small about that result!')


@app.callback(invoke_without_command=True, result_callback=cli_end)
def cli(
    ctx: typer.Context,
    log_level: Annotated[
        str,
        typer.Option(
            '--log-level',
            '-l',
            help='Set the logging level',
            show_choices=True,
            case_sensitive=False,
            autocompletion=lambda: list(LOG_LEVELS.keys()),
            click_type=click.Choice(list(LOG_LEVELS.keys()), case_sensitive=False),
        ),
    ] = 'info',
    add_imports: Annotated[
        list[str],
        typer.Option(
            '--add-imports',
            '-i',
            help='Import additional paths.',
        ),
    ] = [],
) -> None:
    """Main entry point for the TinyGent CLI."""
    setup_logger(log_level)
    logger = logging.getLogger(__name__)

    # Store the start time in the context object
    ctx.obj = ctx.obj or {}
    ctx.obj['start_time'] = time.time()

    logger.info("Starting 'Tinygent' CLI...")
    logger.info('Hold tight, tiny things are about to do huge work!')

    discover_and_register_components(add_imports)
