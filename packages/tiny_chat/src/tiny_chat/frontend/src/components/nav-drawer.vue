<script setup lang='ts'>
import darkAvatar from '@/assets/dark-avatar.png'

const props = defineProps<{
  drawer: boolean
  conversations?: string[]
}>()

const emit = defineEmits<{
  (e: 'update:drawer', value: boolean): void
}>()

const rail = ref(false)
</script>

<template>
  <v-navigation-drawer
    v-model='props.drawer'
    :rail='rail'
    permanent
    app
  >
    <div class='d-flex align-center justify-space-between px-1 py-2' style='height: 64px;'>
      <v-img
        :src='darkAvatar'
        max-width='56'
        max-height='56'
        class='rounded transition-fast-in-fast-out'
        contain
        :style='{ opacity: rail ? 0 : 1, visibility: rail ? "hidden" : "visible" }'
      />

      <v-btn
        icon
        variant='text'
        size='small'
        @click.stop='rail = !rail'
      >
        <v-icon>{{ rail ? 'mdi-chevron-right' : 'mdi-chevron-left' }}</v-icon>
      </v-btn>
    </div>

    <v-list
      nav
      dense
      class='transition-fast-in-fast-out'
      :style='{ opacity: rail ? 0 : 1, visibility: rail ? "hidden" : "visible" }'
    >
      <v-list-subheader>Conversations</v-list-subheader>
      <v-list-item
        v-for='(conv, i) in props.conversations || []'
        :key='i'
        :title='conv'
        prepend-icon='mdi-message-text'
      />
    </v-list>
  </v-navigation-drawer>
</template>
