import { useStateStore } from '@/stores/state-store'

export class WSClient {
  private ws: WebSocket | null = null
  private listeners: ((msg: Message) => void)[] = []
  private serverUrl?: string

  private reconnectTimer: number | null = null
  private readonly RECONNECT_DELAY = 5000

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
    const aa = `${protocol}://${window.location.host}/ws`
    console.log('Resolved WebSocket URL:', aa)
    return aa
  }

  private scheduleReconnect() {
    if (this.reconnectTimer !== null) return
    this.reconnectTimer = window.setInterval(() => {
      if (!this.ws || this.ws.readyState === WebSocket.CLOSED) {
        this.connect()
      }
    }, this.RECONNECT_DELAY)
  }

  private clearReconnect() {
    if (this.reconnectTimer !== null) {
      clearInterval(this.reconnectTimer)
      this.reconnectTimer = null
    }
  }

  connect() {
    let opened = false

    this.clearReconnect()

    this.ws = new WebSocket(this.resolveUrl())

    const { setConnectionStatus } = useStateStore()

    const timeout = window.setTimeout(() => {
      if (opened) return

      console.error('WebSocket connection timeout')
      setConnectionStatus('disconnected')

      try {
        this.ws?.close()
      } catch {}

      this.scheduleReconnect()
    }, 400)

    this.ws.onopen = () => {
      opened = true
      clearTimeout(timeout)

      console.log('WebSocket connection established')
      setConnectionStatus('connected')
      this.clearReconnect()
    }

    this.ws.onclose = () => {
      clearTimeout(timeout)

      console.log('WebSocket connection closed')
      setConnectionStatus('disconnected')
      this.scheduleReconnect()
    }

    this.ws.onerror = (error) => {
      clearTimeout(timeout)

      console.error('WebSocket error:', error)
      setConnectionStatus('disconnected')
      this.scheduleReconnect()
    }

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

  stop() {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      console.warn('Cannot stop, WebSocket is not connected')
      return
    }
    this.ws.send(JSON.stringify({ event: 'stop' }))

    const { setLoadingOwner } = useStateStore()
    setLoadingOwner(null)
  }

  onMessage(callback: (msg: Message) => void) {
    this.listeners.push(callback)
  }
}

export const wsClient = new WSClient()
