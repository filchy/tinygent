<script setup lang="ts">
const props = defineProps<{ msg: UserMessage | AgentTextMessage }>()
</script>

<template>
  <v-sheet
    class='rounded-xl pa-3'
  >
    {{ props.msg.content }}
  </v-sheet>
</template>
