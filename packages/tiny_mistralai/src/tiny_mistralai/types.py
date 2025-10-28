from mistralai import AssistantMessage
from mistralai import SystemMessage
from mistralai import ToolMessage
from mistralai import UserMessage

ChatCompletionMessageParams = (
    AssistantMessage | SystemMessage | UserMessage | ToolMessage
)
