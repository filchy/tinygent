import { useStateStore } from '@/stores/state-store'

export class WSClient {
  private ws: WebSocket | null = null
  private listeners: ((msg: Message) => void)[] = []
  private serverUrl?: string

  constructor(serverUrl?: string) {
    this.serverUrl = serverUrl || import.meta.env.VITE_SERVER_URL
  }

  private resolveUrl(): string {
    if (this.serverUrl) {
      // Replace http/https with ws/wss if user provided full URL
      let url = this.serverUrl.replace(/^http/, 'ws')
      if (!url.endsWith('/ws')) url = `${url}/ws`
      return url
    }

    // Default to current window location
    const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws'
    return `${protocol}://${window.location.host}/ws`
  }

  connect() {
    this.ws = new WebSocket(this.resolveUrl())

    this.ws.onopen = () => {
      console.log('WebSocket connection established')

      const { setConnectionStatus } = useStateStore()
      setConnectionStatus('connected')
    }
    this.ws.onclose = () => {
      console.log('WebSocket connection closed')

      const { setConnectionStatus } = useStateStore()
      setConnectionStatus('disconnected')
    }
    this.ws.onerror = (error) => console.error('WebSocket error:', error)

    this.ws.onmessage = (event) => {
      try {
        const msg: Message = JSON.parse(event.data)
        this.listeners.forEach((callback) => callback(msg))
      } catch (e) {
        console.error('Error parsing WebSocket message:', e)
      }
    }
  }

  send(msg: Message) {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      console.error('WebSocket is not connected')
      return
    }
    this.ws.send(JSON.stringify(msg))
  }

  onMessage(callback: (msg: Message) => void) {
    this.listeners.push(callback)
  }
}

export const wsClient = new WSClient()
