<script setup lang='ts'>
import { ref } from 'vue'
import ChatWindow from './components/chat-window.vue'
import NavigationDrawer from './components/nav-drawer.vue'
import TopBar from './components/top-bar.vue'
import BottomBar from './components/bottom-bar.vue'

const drawer = ref(true)

const conversations = ref(['Conversation 1', 'Conversation 2', 'Conversation 3'])
</script>

<template>
  <v-app class='app-root'>
    <NavigationDrawer v-model:drawer="drawer" :conversations="conversations" />

    <TopBar @toggle-drawer="drawer = !drawer" />

    <v-main class='app-main d-flex flex-column flex-grow-1'>
      <ChatWindow />
    </v-main>

    <BottomBar />
  </v-app>
</template>

<style scoped>
.app-root {
  height: 100vh;          /* full viewport height */
  overflow: hidden;       /* never let v-app scroll */
}

.app-main {
  display: flex;
  flex: 1 1 auto;
  min-height: 0;          /* CRUCIAL: allow children to shrink */
  overflow: hidden;       /* stop v-main scroll */
  flex-direction: column;
}
</style>
