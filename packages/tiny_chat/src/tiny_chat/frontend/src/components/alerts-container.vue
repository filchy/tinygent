<script setup lang="ts">
import { useStateStore } from '@/stores/state-store'

const { connectionStatus } = useStateStore()
</script>

<template>
  <div style="width: 100%; max-width: min(48rem, 100vw); margin: 0 auto">
    <v-alert
      v-if="connectionStatus === 'disconnected'"
      type="error"
      text="Could not reach the server. Please check your connection."
      class="pb-2"
    />
  </div>
</template>
