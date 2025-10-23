from tiny_brave.tools import brave_images_search
from tiny_brave.tools import brave_news_search
from tiny_brave.tools import brave_videos_search
from tiny_brave.tools import brave_web_search

from tinygent.tools.tool import tool


def _register_tools() -> None:
    tool(hidden=False)(brave_news_search)
    tool(hidden=False)(brave_web_search)
    tool(hidden=False)(brave_images_search)
    tool(hidden=False)(brave_videos_search)


_register_tools()
