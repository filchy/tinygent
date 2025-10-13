from tinygent.llms.openai import OpenAIConfig
from tinygent.llms.openai import OpenAILLM
from tinygent.runtime.global_registry import GlobalRegistry

GlobalRegistry().get_registry().register_llm('openai', OpenAIConfig, OpenAILLM)
