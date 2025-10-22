from tinygent.tools.tool import tool

from tiny_brave.tools.news_search import brave_news_search


def _register_tools() -> None:
    tool(hidden=False)(brave_news_search)


_register_tools()
