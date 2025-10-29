<script setup lang='ts'>
import { ref, watch } from 'vue'
import { useDisplay } from 'vuetify'
import darkAvatar from '@/assets/dark-avatar.png'

const props = defineProps<{
  drawer: boolean
  conversations?: string[]
}>()

const emit = defineEmits<{
  (e: 'update:drawer', value: boolean): void
}>()

const localDrawer = ref(props.drawer)
watch(() => props.drawer, (val) => (localDrawer.value = val))
watch(localDrawer, (val) => emit('update:drawer', val))

const rail = ref(false)
const { smAndDown } = useDisplay()

watch(smAndDown, (val) => {
  if (val) {
    localDrawer.value = false
  } else {
    localDrawer.value = true
  }
}, { immediate: true })
</script>

<template>
  <v-navigation-drawer
    v-model='localDrawer'
    :rail='!smAndDown && rail'
    :temporary='smAndDown'
    :permanent='!smAndDown'
    app
  >
    <div class='d-flex align-center justify-space-between px-1 py-2' style='height: 64px;'>
      <v-img
        :src='darkAvatar'
        max-width='56'
        max-height='56'
        class='rounded transition-fast-in-fast-out'
        contain
        :style='{ opacity: (!smAndDown && rail) ? 0 : 1, visibility: (!smAndDown && rail) ? "hidden" : "visible" }'
      />

      <v-btn
        icon
        variant='text'
        size='small'
        @click.stop='smAndDown ? (localDrawer = false) : (rail = !rail)'
      >
        <v-icon>{{ smAndDown ? 'mdi-close' : (rail ? 'mdi-chevron-right' : 'mdi-chevron-left') }}</v-icon>
      </v-btn>
    </div>

    <v-list
      nav
      dense
      class='transition-fast-in-fast-out'
      :style='{ opacity: (!smAndDown && rail) ? 0 : 1, visibility: (!smAndDown && rail) ? "hidden" : "visible" }'
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
