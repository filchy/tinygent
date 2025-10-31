const messages = ref<Message[]>([])

export function useChatStore() {
  const addMessage = (msg: Message) => {
    messages.value.push(msg)
  }

  const clearMessages = () => {
    messages.value = []
  }

  return {
    messages,
    addMessage,
    clearMessages,
  }
}
