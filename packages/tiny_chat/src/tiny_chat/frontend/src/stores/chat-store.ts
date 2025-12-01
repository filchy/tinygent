import { useStateStore } from '@/stores/state-store'

const messages = ref<Message[]>([])
const chatId = ref<string>(crypto.randomUUID())
const { loadingOwner, setLoadingOwner } = useStateStore()

export function useChatStore() {
  const addMessage = (msg: Message) => {
    const last = messages.value[messages.value.length - 1]

    // Automatically update loading owner:
    if (msg.sender === 'user') {
      setLoadingOwner('agent')
    } else if (msg.sender === 'agent') {
      setLoadingOwner('user')
    }

    // If agent finishes sending, clear loading
    if (msg.type !== 'loading' && msg.sender === 'agent') {
      setLoadingOwner(null)
    }

    // Replace existing loading message
    if (last && last.type === 'loading' && msg.type !== 'loading') {
      messages.value.pop()
    }

    // Handle streaming updates
    const existing = messages.value.find((m: Message) => m.id === msg.id)

    if (existing) {
      existing.content += msg.content
      return
    }

    messages.value.push(msg)
  }

  const clearMessages = () => {
    messages.value = []
  }

  watch(loadingOwner, (owner) => {
    const last = messages.value[messages.value.length - 1]
    const hasLoading = last?.type === 'loading'

    if (owner && owner === 'agent' && !hasLoading) {
      messages.value.push({
        id: `loading-agent-${crypto.randomUUID()}`,
        type: 'loading',
        sender: owner,
        content: '',
      })
    } else if (!owner && hasLoading) {
      messages.value.pop()
    }
  })

  return {
    messages,
    chatId,
    addMessage,
    clearMessages,
  }
}
