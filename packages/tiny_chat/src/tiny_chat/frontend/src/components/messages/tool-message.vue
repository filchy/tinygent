<script setup lang="ts">
const props = defineProps<{ msg: ToolMessage }>()
</script>

<template>
  <v-sheet
    class='rounded-xl pa-3'
  >
    {{ props.msg.tool_name }}
  </v-sheet>
</template>
