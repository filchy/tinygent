from tiny_brave.tools import brave_images_search
from tiny_brave.tools import brave_news_search
from tiny_brave.tools import brave_videos_search
from tiny_brave.tools import brave_web_search
from tinygent.tools.tool import register_tool


def _register_tools() -> None:
    register_tool(hidden=False)(brave_news_search)
    register_tool(hidden=False)(brave_web_search)
    register_tool(hidden=False)(brave_images_search)
    register_tool(hidden=False)(brave_videos_search)


_register_tools()
