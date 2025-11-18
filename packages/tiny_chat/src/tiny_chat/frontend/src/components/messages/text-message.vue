<script setup lang="ts">
const props = defineProps<{ msg: MainMessage }>()
</script>

<template>
  <v-sheet class="rounded-xl pa-3">
    {{ props.msg.content }}
  </v-sheet>
</template>
