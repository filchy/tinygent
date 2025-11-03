const messages = ref<Message[]>([])

export function useChatStore() {
  const addMessage = (msg: Message) => {
    const last = messages.value[messages.value.length - 1]

    if (last && last.type === 'loading' && msg.type !== 'loading') {
      messages.value.pop()
    }

    if (msg.streaming) {
      const existing = messages.value.find(
        (m) => m.id === msg.id
      )
      if (existing) {
        existing.content += msg.content
        return
      }
    }

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
