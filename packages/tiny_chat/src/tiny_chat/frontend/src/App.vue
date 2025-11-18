<script setup lang="ts">
import Alerts from '@/components/alerts-container.vue'
import ChatWindow from '@/components/chat-window.vue'
import NavigationDrawer from '@/components/nav-drawer.vue'
import InfoDrawer from '@/components/info-drawer.vue'
import TopBar from '@/components/top-bar.vue'
import BottomBar from '@/components/bottom-bar.vue'

import { wsClient } from '@/services/ws-client'
import { useChatStore } from '@/stores/chat-store'

const { addMessage } = useChatStore()

onMounted(() => {
  console.log('Connecting to WebSocket server...')
  wsClient.connect()

  // addMessage({
  //   id: crypto.randomUUID(),
  //   type: 'text',
  //   sender: 'bot',
  //   content: `
  // Bubble sort je jednoduchý algoritmus pro třídění, který opakovaně prochází seznam, porovnává sousední prvky a vyměňuje je, pokud jsou ve špatném pořadí. Tento proces se opakuje, dokud není seznam seřazen.
  //
  // Zde je jednoduchá implementace bubble sort v Pythonu:
  //
  // \`\`\`python
  // def bubble_sort(arr):
  //     n = len(arr)
  //     for i in range(n):
  //         for j in range(0, n - i - 1):
  //             if arr[j] > arr[j + 1]:
  //                 arr[j], arr[j + 1] = arr[j + 1], arr[j]
  //     return arr
  // \`\`\`
  //
  // Tato funkce \`bubble_sort\` přijímá seznam \`arr\` jako vstup a třídí ho vzestupně pomocí algoritmu bubble sort.
  //
  // Pokud máte další dotazy nebo potřebujete další informace, dejte mi vědět!
  // `,
  // })

  wsClient.onMessage((msg) => {
    addMessage(msg)
  })
})

const drawer = ref(true)

const conversations = ref(['Conversation 1', 'Conversation 2', 'Conversation 3'])

const sendMessage = (message: string) => {
  wsClient.send({
    id: crypto.randomUUID(),
    type: 'text',
    sender: 'user',
    content: message,
  } as UserMessage)
}
</script>

<template>
  <v-app class="app-root">
    <NavigationDrawer v-model:drawer="drawer" :conversations="conversations" />
    <InfoDrawer />

    <TopBar @toggle-drawer="drawer = !drawer" />

    <v-main class="app-main d-flex flex-column flex-grow-1">
      <div class="d-flex flex-column" style="position: relative; height: 100%">
        <Alerts />
        <ChatWindow />
      </div>
    </v-main>

    <BottomBar @send-message="sendMessage" />
  </v-app>
</template>

<style scoped>
.app-root {
  height: 100vh;
  overflow: hidden;
}

.app-main {
  display: flex;
  flex: 1 1 auto;
  min-height: 0;
  overflow: hidden;
  flex-direction: column;
}
</style>
