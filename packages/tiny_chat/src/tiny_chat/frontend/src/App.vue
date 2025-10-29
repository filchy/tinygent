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
  <v-app>
    <NavigationDrawer v-model:drawer="drawer" :conversations="conversations" />

    <TopBar @toggle-drawer="drawer = !drawer" />

    <v-main class='app-main d-flex flex-column flex-grow-1'>
      <ChatWindow class='flex-grow-1'/>
    </v-main>

    <BottomBar />
  </v-app>
</template>

<style scoped>
.app-main {
  min-height: 0;
  overflow: hidden;
}
</style>
