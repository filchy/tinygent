<script setup lang="ts">
import { useTheme } from 'vuetify'

import darkAvatar from '@/assets/dark-avatar.png'
import lightAvatar from '@/assets/light-avatar.png'

const theme = useTheme()

const props = defineProps<{ msg: BaseMessage }>()

const currentAvatar = computed(() =>
  theme.global.current.value.dark ? lightAvatar : darkAvatar
)
</script>

<template>
  <div
    class='d-flex chat-message'
    :class="props.msg.sender === 'user' ? 'justify-end' : 'justify-start'"
  >
    <v-avatar size='36' class='mr-2 flex-shrink-0' v-if='props.msg.sender === "agent"'>
      <v-img :src="currentAvatar" />
    </v-avatar>

    <v-sheet max-width='70%' color='transparent'>
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
