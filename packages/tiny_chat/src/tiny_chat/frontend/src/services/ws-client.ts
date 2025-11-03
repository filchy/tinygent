import { useStateStore } from '@/stores/state-store'

export class WSClient {
  private ws: WebSocket | null = null
  private listeners: ((msg: Message) => void)[] = []
  private serverUrl?: string

  private reconnectTimer: number | null = null
  private heartbeatTimer: number | null = null
  private readonly RECONNECT_DELAY = 5000
  private readonly HEARTBEAT_INTERVAL = 25000

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

  private startHeartbeat() {
    this.stopHeartbeat()
    this.heartbeatTimer = window.setInterval(() => {
      if (this.ws && this.ws.readyState === WebSocket.OPEN) {
        try {
          this.ws.send(JSON.stringify({ type: 'ping' }))
        } catch {}
      }
    }, this.HEARTBEAT_INTERVAL)
  }

  private stopHeartbeat() {
    if (this.heartbeatTimer !== null) {
      clearInterval(this.heartbeatTimer)
      this.heartbeatTimer = null
    }
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
    this.clearReconnect()
    this.stopHeartbeat()

    this.ws = new WebSocket(this.resolveUrl())

    const { setConnectionStatus } = useStateStore()

    this.ws.onopen = () => {
      console.log('WebSocket connection established')

      setConnectionStatus('connected')
      this.clearReconnect()
      this.startHeartbeat()
    }

    this.ws.onclose = () => {
      console.log('WebSocket connection closed')

      setConnectionStatus('disconnected')
      this.scheduleReconnect()
      this.stopHeartbeat()
    }

    this.ws.onerror = (error) => {
      console.error('WebSocket error:', error)

      setConnectionStatus('disconnected')
      this.scheduleReconnect()
      this.stopHeartbeat()
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

  onMessage(callback: (msg: Message) => void) {
    this.listeners.push(callback)
  }
}

export const wsClient = new WSClient()
