from tinygent.datamodels.messages import BaseMessage


def is_final_answer(message: BaseMessage) -> bool:
    """Check if the message is marked as the final answer."""
    return message.metadata.get('is_final_answer', False)
