def render_template(template_str: str, context: dict) -> str:
    from jinja2 import BaseLoader
    from jinja2 import Environment

    env = Environment(loader=BaseLoader(), trim_blocks=True, lstrip_blocks=True)
    return env.from_string(template_str).render(**context)
