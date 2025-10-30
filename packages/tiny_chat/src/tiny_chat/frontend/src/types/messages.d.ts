declare interface BaseMessage {
  id: string
  type: 'text' | 'reasoning' | 'debug' | 'delta'
  sender: 'user' | 'agent'
  content: string
  streaming?: boolean // true if partial / still incoming
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

declare type Message = UserMessage | AgentTextMessage | AgentReasoningMessage
