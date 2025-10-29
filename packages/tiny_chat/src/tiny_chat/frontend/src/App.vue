<script setup lang='ts'>
import { ref } from 'vue'
import NavigationDrawer from './components/nav-drawer.vue'
import TopBar from './components/top-bar.vue'

const drawer = ref(true)

const conversations = ref([
  'Conversation 1',
  'Conversation 2',
  'Conversation 3'
])
</script>

<template>
  <v-app>
    <NavigationDrawer v-model:drawer='drawer' :conversations='conversations' />

    <TopBar @toggle-drawer='drawer = !drawer' />
  </v-app>
</template>
