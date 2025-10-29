<script setup lang='ts'>
import { ref, nextTick } from 'vue'

type Message =
  | { type: 'text'; sender: 'user' | 'bot'; text: string }
  | { type: 'image'; sender: 'bot'; url: string; caption?: string }
  | { type: 'error'; text: string }

const baseMessages: Message[] = [
    { type: 'text', sender: 'bot', text: 'Hello! How can I help you?' },
    { type: 'text', sender: 'user', text: 'Show me a picture' },
]

const messages = ref<Message[]>([])

for (let i = 0; i < 10; i++) {
    messages.value.push(...baseMessages)
}

const newMessage = ref('')
const chatRef = ref<HTMLDivElement>()

function scrollToBottom() {
  nextTick(() => {
    if (chatRef.value) {
      chatRef.value.scrollTop = chatRef.value.scrollHeight
    }
  })
}
</script>

<template>
<div
  class="d-flex flex-column flex-grow-1 pa-0 chat-container"
>
  <div
    ref="chatRef"
    class="flex-grow-1 pa-4 chat-scroll"
  >
    <div
      v-for="(msg, i) in messages"
      :key="i"
      class="d-flex"
      :class="msg.sender === 'user' ? 'justify-end' : 'justify-start'"
    >
      <v-sheet
        class="pa-3"
        :color="msg.sender === 'user' ? 'primary' : 'grey-lighten-2'"
        rounded
        max-width="70%"
      >
        <span :class="msg.sender === 'user' ? 'text-white' : ''">
          {{ msg.text }}
        </span>
      </v-sheet>
    </div>
  </div>
</div>
</template>


<style scoped>
.chat-container {
  min-height: 0;
  flex: 1 1 auto;
}

.chat-scroll {
  flex: 1 1 auto;
  min-height: 0;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 8px;

  width: 100%;
  max-width: min(48rem, 100vw);
  margin-left: auto;
  margin-right: auto;
}
</style>
