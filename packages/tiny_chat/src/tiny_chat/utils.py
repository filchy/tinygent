from typing import cast

import tiny_chat.message as ChatMessages
import tinygent.datamodels.messages as TinyMessages


def __map_user_msg(
    tc_message: ChatMessages.UserMessage,
) -> TinyMessages.TinyHumanMessage:
    return TinyMessages.TinyHumanMessage(content=tc_message.content)


def __map_agent_msg(
    tc_message: ChatMessages.AgentMessage,
) -> TinyMessages.TinyChatMessage:
    return TinyMessages.TinyChatMessage(content=tc_message.content)


def __map_tool_call_msg(
    tc_message: ChatMessages.AgentToolCallMessage,
) -> TinyMessages.TinyToolCall:
    tc = TinyMessages.TinyToolCall(
        tool_name=tc_message.tool_name,
        arguments=tc_message.tool_args,
    )
    tc.result = tc_message.content
    return tc


def tinychat_2_tinygent_message(
    tc_message: ChatMessages.BaseMessage,
) -> TinyMessages.AllTinyMessages:
    match type(tc_message):
        case ChatMessages.UserMessage:
            return __map_user_msg(cast(ChatMessages.UserMessage, tc_message))
        case ChatMessages.AgentMessage:
            return __map_agent_msg(cast(ChatMessages.AgentMessage, tc_message))
        case ChatMessages.AgentToolCallMessage:
            return __map_tool_call_msg(
                cast(ChatMessages.AgentToolCallMessage, tc_message)
            )
        case _:
            raise ValueError(f'Unsupported message type: {type(tc_message)}')
