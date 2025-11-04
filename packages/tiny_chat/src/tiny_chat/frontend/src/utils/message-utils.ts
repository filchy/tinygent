export function isChildMessage(msg: Message): msg is ChildMessage {
  return 'parent_id' in msg
}

export function isMainMessage(msg: Message): msg is MainMessage {
  return !isChildMessage(msg)
}

export function isUserMessage(msg: Message): msg is UserMessage {
  return msg.sender === 'user'
}
