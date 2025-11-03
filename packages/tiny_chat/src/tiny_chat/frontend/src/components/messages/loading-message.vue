<script setup lang="ts">
const props = defineProps<{ msg: LoadingMessage }>()
</script>

<template>
  <v-sheet
    class='pa-0 rounded-xl'
  >
    <div class="d-flex align-center justify-center px-4 py-1 text-medium-emphasis">
      <span class="wave-dots">
        <span>.</span><span>.</span><span>.</span>
      </span>
    </div>
  </v-sheet>
</template>

<style scoped>
.wave-dots {
  display: inline-flex;
  gap: 4px;
  font-size: 1.4rem;
}

.wave-dots span {
  display: inline-block;
  animation: wave 1.2s infinite ease-in-out;
}

.wave-dots span:nth-child(2) {
  animation-delay: 0.2s;
}

.wave-dots span:nth-child(3) {
  animation-delay: 0.4s;
}

@keyframes wave {
  0%, 60%, 100% {
    transform: translateY(0);
    opacity: 0.6;
  }
  30% {
    transform: translateY(-6px);
    opacity: 1;
  }
}
</style>
