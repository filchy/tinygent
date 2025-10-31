<script setup lang="ts">
import { wsClient } from '@/services/ws-client'
import { useChatStore } from '@/stores/chat-store'

const { messages } = useChatStore()
const chatRef = ref<HTMLDivElement>()

const scrollToBottom = () => {
  if (chatRef.value) {
    chatRef.value.scrollTop = chatRef.value.scrollHeight
  }
}

onMounted(() => {
  wsClient.onMessage(() => {
    scrollToBottom()
  })
})
</script>

<template>
  <div class="chat-container">
    <div ref="chatRef" class="chat-scroll">
      <div
        v-for="(msg, i) in messages"
        :key="i"
        class="d-flex chat-message"
        :class="msg.sender === 'user' ? 'justify-end' : 'justify-start'"
      >
        <v-sheet
          class="pa-3"
          :color="msg.sender === 'user' ? 'primary' : 'grey-lighten-2'"
          rounded
          max-width="70%"
        >
          <span :class="msg.sender === 'user' ? 'text-white' : ''">
            {{ msg.content }}
          </span>
        </v-sheet>
      </div>
    </div>
  </div>
</template>

<style scoped>
.chat-container {
  position: relative;
  display: flex;
  flex: 1 1 auto;
  min-height: 0;
}

.chat-scroll {
  flex: 1 1 auto;
  min-height: 0;
  overflow-y: auto;
  padding: 16px;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;

  width: 100%;
  margin: 0 auto;
}

.chat-message {
  width: 100%;
  max-width: min(48rem, 100vw);
}
</style>
