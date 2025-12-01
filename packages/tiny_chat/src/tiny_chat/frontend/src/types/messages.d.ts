declare interface BaseMessage {
  id: string
  chat_id: string
  type: 'text' | 'reasoning' | 'loading' | 'source' | 'tool'
  sender: Role
  content: string
}

// Main messages (those which can be standalone)
declare interface UserMessage extends BaseMessage {
  type: 'text'
  sender: 'user'
}

declare interface AgentMessage extends BaseMessage {
  type: 'text'
  sender: 'agent'
}

declare interface LoadingMessage extends BaseMessage {
  type: 'loading'
}

// Child messages (needs parent)
declare interface ChildMessage extends BaseMessage {
  parent_id: string
}

declare interface ReasoningMessage extends ChildMessage {
  type: 'reasoning'
  sender: 'agent'
}

declare interface ToolMessage extends ChildMessage {
  type: 'tool'
  sender: 'agent'

  content: string
  tool_name: string
  tool_args: Record<string, any>
}

declare interface SourceMessage extends ChildMessage {
  type: 'source'
  sender: 'agent'

  url: string
  name: string
  favicon?: string
  description?: string
}

// Union type for all messages
declare type Message =
  | UserMessage
  | AgentTextMessage
  | LoadingMessage
  | ReasoningMessage
  | SourceMessage
  | ToolMessage

// Main messages union
declare type MainMessage = UserMessage | AgentMessage | LoadingMessage

// Child messages union
declare type ChildMessage = ReasoningMessage | SourceMessage | ToolMessage

// Message group with main message and optional child messages
declare interface MessageGroup {
  group_id: string

  main?: MainMessage
  children?: ChildMessage[]
}
