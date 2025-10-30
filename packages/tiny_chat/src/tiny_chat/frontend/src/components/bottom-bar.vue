<script setup lang="ts">
import { useTheme } from 'vuetify'
import darkAvatar from '@/assets/dark-avatar.png'
import lightAvatar from '@/assets/light-avatar.png'

const emit = defineEmits<{
  (e: 'send-message', message: string): void
}>()

const theme = useTheme()
const message = ref('')

const currentAvatar = computed(() => (theme.global.current.value.dark ? lightAvatar : darkAvatar))
</script>

<template>
  <v-footer
    app
    class="d-flex flex-column align-center justify-center text-caption font-weight-thin"
    color="transparent"
  >
    <v-text-field
      v-model="message"
      label="Type your tiny message here..."
      width="100%"
      max-width="min(48rem, 100vw)"
      variant="solo"
      color="grey"
      rounded
    >
      <template #append-inner>
        <v-tooltip bottom>
          <template #activator="{ props }">
            <v-btn
              icon
              variant="text"
              v-bind="props"
              :disabled="!message.trim()"
              @click='emit("send-message", message); message = ""'
            >
              <v-icon>mdi-send</v-icon>
            </v-btn>
          </template>
          Send Message
        </v-tooltip>
      </template>
    </v-text-field>

    <span class="d-flex align-center text-caption">
      Build with
      <v-img :src="currentAvatar" alt="tinygent logo" width="32" height="32" class="mx-1" contain />
      <a
        class="font-weight-bold"
        href="https://github.com/filchy/tinygent"
        target="_blank"
        style="text-decoration: none; color: inherit"
      >
        tinygent
      </a>
    </span>
  </v-footer>
</template>
