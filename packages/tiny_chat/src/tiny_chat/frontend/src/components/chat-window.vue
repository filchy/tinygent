<script setup lang="ts">
import { useTheme } from 'vuetify'

import Message from './message.vue'

import { wsClient } from '@/services/ws-client'
import { useChatStore } from '@/stores/chat-store'
import darkAvatar from '@/assets/dark-avatar.png'
import lightAvatar from '@/assets/light-avatar.png'

const { messages, addMessage } = useChatStore()
const chatRef = ref<HTMLDivElement>()
const theme = useTheme()

const currentAvatar = computed(() =>
  theme.global.current.value.dark ? lightAvatar : darkAvatar
)

const scrollToBottom = () => {
  if (chatRef.value) chatRef.value.scrollTop = chatRef.value.scrollHeight
}

onMounted(() => {
  wsClient.onMessage(() => scrollToBottom())
})
</script>

<template>
  <div class="chat-container">
    <div ref="chatRef" class="chat-scroll">
      <Message
        v-for="(msg, i) in messages"
        :key="i"
        :msg="msg"
      />
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
  gap: 12px;
  width: 100%;
  margin: 0 auto;
}

.chat-message {
  width: 100%;
  max-width: min(48rem, 100vw);
  display: flex;
  align-items: flex-end;
}
</style>
