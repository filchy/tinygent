<script setup lang="ts">
import { useDisplay } from 'vuetify'
import { useStateStore } from '@/stores/state-store'

const emit = defineEmits<{
  (e: 'toggle-drawer'): void
}>()

const { smAndDown } = useDisplay()
const { connectionStatus } = useStateStore()
</script>

<template>
  <v-app-bar app flat density='comfortable' color='transparent'>
    <v-app-bar-nav-icon v-if="smAndDown" @click="emit('toggle-drawer')" />
  </v-app-bar>
</template>
