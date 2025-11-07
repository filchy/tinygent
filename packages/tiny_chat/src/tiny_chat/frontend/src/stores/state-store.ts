type ConnectionStatus = 'connected' | 'disconnected' | 'init'

const connectionStatus = ref<ConnectionStatus>('init')
const loadingOwner = ref<Role | null>(null)

export function useStateStore() {
  const setConnectionStatus = (status: ConnectionStatus) => {
    connectionStatus.value = status
  }

  const setLoadingOwner = (owner: Role | null) => {
    loadingOwner.value = owner
  }

  return {
    connectionStatus,
    loadingOwner,
    setConnectionStatus,
    setLoadingOwner,
  }
}
