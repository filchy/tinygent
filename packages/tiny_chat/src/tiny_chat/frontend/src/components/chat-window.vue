<script setup lang="ts">
import { useTheme } from 'vuetify'

import Message from './message.vue'
import MessageGroup from './message-group.vue'

import {
  isChildMessage,
  isMainMessage,
  isUserMessage
} from '@/utils/message-utils'
import { wsClient } from '@/services/ws-client'
import { useChatStore } from '@/stores/chat-store'
import { useStateStore } from '@/stores/state-store'
import darkAvatar from '@/assets/dark-avatar.png'
import lightAvatar from '@/assets/light-avatar.png'

const { connectionStatus } = useStateStore()
const { messages, addMessage } = useChatStore()
const chatRef = ref<HTMLDivElement>()
const theme = useTheme()

const currentAvatar = computed(() =>
  theme.global.current.value.dark ? lightAvatar : darkAvatar
)

const messageGroups = computed(() => {
  const groups: MessageGroup[] = []

  for (const msg of messages.value) {
    const groupId = isChildMessage(msg) ? msg.parent_id : msg.id

    let group = groups.find(g => g.group_id === groupId)

    if (!group) {
      group = { group_id: groupId, main: undefined, children: [] }
      groups.push(group)
    }

    if (isMainMessage(msg)) {
      group.main = msg
    } else if (isChildMessage(msg)) {
      group.children = group.children ?? []
      group.children.push(msg)
    }
  }

  return groups
})
</script>

<template>
  {{ messageGroups }}
  <div class='chat-container' ref='chatRef'>
    <div class='chat-column' style='gap: 12px;'>
      <span
        v-for='(group, index) in messageGroups'
        :key='group.main?.id ?? `no-main-${index}`'
      >
        <Message
          v-if='group.main && isUserMessage(group.main)'
          :msg='group.main'
        />

        <MessageGroup
          v-else
          class='align-end'
          :message-group='group'
        />
      </span>
    </div>
  </div>
</template>

<style scoped>
.chat-container {
  position: relative;
  display: flex;
  justify-content: center;
  flex: 1 1 auto;
  min-height: 0;
  height: 100%;
  overflow-y: auto;
  align-items: flex-start;
  gap: 12px;
  width: 100%;
  margin: 0 auto;
}

.chat-column {
  position: relative;
  max-width: min(48rem, 100%);
  width: 100%;
  display: flex;
  flex-direction: column;
}
</style>
