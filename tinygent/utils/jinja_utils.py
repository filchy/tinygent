from typing import Any

from jinja2 import BaseLoader
from jinja2 import Environment
from jinja2 import meta


def validate_template(template_str: str, required_fields: set[str]) -> bool:
    env = Environment(loader=BaseLoader(), trim_blocks=True, lstrip_blocks=True)
    ast = env.parse(template_str)
    found_fields = meta.find_undeclared_variables(ast)
    return required_fields.issubset(found_fields)


def render_template(template_str: str, context: dict[str, Any]) -> str:
    env = Environment(loader=BaseLoader(), trim_blocks=True, lstrip_blocks=True)
    return env.from_string(template_str).render(**context)
