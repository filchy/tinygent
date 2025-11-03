<script setup lang="ts">
import { useTheme } from 'vuetify'
import darkAvatar from '@/assets/dark-avatar.png'
import lightAvatar from '@/assets/light-avatar.png'

import { useChatStore } from '@/stores/chat-store'
import { useStateStore } from '@/stores/state-store'

const emit = defineEmits<{
  (e: 'send-message', message: string): void
}>()

const theme = useTheme()
const message = ref('')
const { addMessage } = useChatStore()
const { loadingOwner, connectionStatus, setLoadingOwner } = useStateStore()

let typingWatchEnabled = true

watch(message, (val) => {
  if (!typingWatchEnabled) return

  if (val.trim().length > 0) {
    setLoadingOwner('user')
  } else {
    setLoadingOwner(null)
  }
})

const currentAvatar = computed(() => (theme.global.current.value.dark ? lightAvatar : darkAvatar))

const sendMessageEnabled = computed(
  () => message.value.trim().length > 0 && loadingOwner.value !== 'agent' &&  connectionStatus.value
  !== 'disconnected'
)

const addUserMessage = (msg: string) => {
  addMessage({
    id: crypto.randomUUID(),
    type: 'text',
    sender: 'user',
    content: msg,
  })
}

const sendMessage = () => {
  const messageValue = message.value.trim()
  if (!messageValue) return

  typingWatchEnabled = false

  message.value = ''

  addUserMessage(messageValue)
  emit('send-message', messageValue)

  nextTick(() => {
    typingWatchEnabled = true
  })
}
</script>

<template>
  <v-footer
    app
    class='d-flex flex-column align-center justify-center text-caption font-weight-thin'
    color='transparent'
  >
    <v-text-field
      v-model='message'
      label='Type your tiny message here...'
      width='100%'
      max-width='min(48rem, 100vw)'
      variant='solo'
      color='grey'
      autocomplete='off'
      rounded
      @keyup.enter='sendMessage'
    >
      <template #append-inner>
        <v-tooltip bottom>
          <template #activator='{ props }'>
            <v-btn
              icon
              variant='text'
              v-bind='props'
              :disabled='!sendMessageEnabled'
              @click='sendMessage'
            >
              <v-icon>mdi-send</v-icon>
            </v-btn>
          </template>
          Send Message
        </v-tooltip>
      </template>
    </v-text-field>

    <span class='d-flex align-center text-caption'>
      Build with
      <v-img :src='currentAvatar' alt='tinygent logo' width='32' height='32' class='mx-1' contain />
      <a
        class='font-weight-bold'
        href='https://github.com/filchy/tinygent'
        target='_blank'
        style='text-decoration: none; color: inherit'
      >
        tinygent
      </a>
    </span>
  </v-footer>
</template>
