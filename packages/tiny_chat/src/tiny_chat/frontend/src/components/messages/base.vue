<script setup lang="ts">
const props = defineProps<{ msg: BaseMessage }>()
</script>

<template>
  <div
    class="d-flex chat-message align-start"
    :class="props.msg.sender === 'user' ? 'justify-end' : 'justify-start'"
  >
    <v-sheet max-width="70%" color="transparent">
      <slot />
    </v-sheet>
  </div>
</template>

<style scoped>
.chat-message {
  width: 100%;
  display: flex;
  align-items: flex-end;
  max-width: min(48rem, 100vw);
  width: 100%;
}
</style>
