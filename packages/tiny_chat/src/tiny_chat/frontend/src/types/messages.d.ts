declare interface BaseMessage {
  id: string
  type: 'text' | 'reasoning' | 'loading',
  sender: 'user' | 'agent'
  content: string
  streaming?: boolean // true if partial / still incoming
}

declare interface LoadingMessage extends BaseMessage {
  type: 'loading'
}

declare interface UserMessage extends BaseMessage {
  type: 'text'
  sender: 'user'
}

declare interface AgentTextMessage extends BaseMessage {
  type: 'text'
  sender: 'agent'
}

declare interface AgentReasoningMessage extends BaseMessage {
  type: 'reasoning'
  sender: 'agent'
}

declare type Message = LoadingMessage | UserMessage | AgentTextMessage | AgentReasoningMessage
