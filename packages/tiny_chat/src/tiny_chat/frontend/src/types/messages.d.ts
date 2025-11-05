declare interface BaseMessage {
  id: string
  type: 'text' | 'reasoning' | 'loading' | 'sources' | 'tool'
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
  content: string = ''
  tool_name: string
  tool_args: Record<string, any>
}

declare interface SourcesMessage extends ChildMessage {
  type: 'sources',
  sender: 'agent'
}

// Union type for all messages
declare type Message = UserMessage | AgentTextMessage | LoadingMessage | ReasoningMessage | SourcesMessage | ToolMessage

// Main messages union
declare type MainMessage = UserMessage | AgentMessage | LoadingMessage

// Child messages union
declare type ChildMessage = ReasoningMessage | SourcesMessage | ToolMessage

// Message group with main message and optional child messages
declare interface MessageGroup {
  group_id: string

  main?: MainMessage
  children?: ChildMessage[]
}
