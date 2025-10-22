from tiny_brave.tools import brave_news_search
from tiny_brave.tools import brave_web_search
from tinygent.tools.tool import tool


def _register_tools() -> None:
    tool(hidden=False)(brave_news_search)
    tool(hidden=True)(brave_web_search)


_register_tools()
