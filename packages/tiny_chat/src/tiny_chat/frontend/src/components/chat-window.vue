<script setup lang="ts">
import { ref } from 'vue'

// message model
type Message =
  | { type: 'text'; sender: 'user' | 'bot'; text: string }
  | { type: 'image'; sender: 'bot'; url: string; caption?: string }
  | { type: 'error'; text: string }

const messages = ref<Message[]>([
  { type: 'text', sender: 'bot', text: 'Hello! How can I help you?' },
  { type: 'text', sender: 'user', text: 'Show me a picture' },
])

const newMessage = ref('')

function sendMessage() {
  if (!newMessage.value.trim()) return
  messages.value.push({ type: 'text', sender: 'user', text: newMessage.value })
  newMessage.value = ''

  // fake bot reply
  setTimeout(() => {
    messages.value.push({ type: 'text', sender: 'bot', text: 'Got it! (mock reply)' })
  }, 600)
}
</script>

<template>
  <v-container
    fluid
    class='flex-grow-1 d-flex flex-column pa-4'
    style='gap: 8px;'
  >
    ahoj
  </v-container>
</template>
