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

    <v-app>
      <ChatWindow />
    </v-app>

    <BottomBar />
  </v-app>
</template>
