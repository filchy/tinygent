<script setup lang="ts">
import { useDisplay, useTheme } from 'vuetify'
import darkAvatar from '@/assets/dark-avatar.png'
import lightAvatar from '@/assets/light-avatar.png'

const props = defineProps<{
  drawer: boolean
  conversations?: string[]
}>()

const emit = defineEmits<{
  (e: 'update:drawer', value: boolean): void
}>()

const localDrawer = ref(props.drawer)
watch(
  () => props.drawer,
  (val) => (localDrawer.value = val),
)
watch(localDrawer, (val) => emit('update:drawer', val))

const rail = ref(false)
const { smAndDown } = useDisplay()

watch(
  smAndDown,
  (val) => {
    if (val) {
      localDrawer.value = false
    } else {
      localDrawer.value = true
    }
  },
  { immediate: true },
)

const theme = useTheme()
const isDark = ref(theme.global.current.value.dark)

watch(isDark, (val) => {
  theme.global.name.value = val ? 'dark' : 'light'
})

const currentAvatar = computed(() => (isDark.value ? lightAvatar : darkAvatar))
</script>

<template>
  <v-navigation-drawer
    v-model="localDrawer"
    :rail="!smAndDown && rail"
    :temporary="smAndDown"
    :permanent="!smAndDown"
    app
  >
    <!-- Header -->
    <div class="d-flex align-center justify-space-between px-1 py-2" style="height: 64px">
      <v-img
        :src="currentAvatar"
        max-width="56"
        max-height="56"
        class="rounded transition-fast-in-fast-out"
        contain
        :style="{
          opacity: !smAndDown && rail ? 0 : 1,
          visibility: !smAndDown && rail ? 'hidden' : 'visible',
        }"
      />

      <v-btn
        icon
        variant="text"
        size="small"
        @click.stop="smAndDown ? (localDrawer = false) : (rail = !rail)"
      >
        <v-icon>{{
          smAndDown ? 'mdi-close' : rail ? 'mdi-chevron-right' : 'mdi-chevron-left'
        }}</v-icon>
      </v-btn>
    </div>

    <!-- Conversation list -->
    <v-list
      nav
      dense
      class="transition-fast-in-fast-out flex-grow-1"
      :style="{
        opacity: !smAndDown && rail ? 0 : 1,
        visibility: !smAndDown && rail ? 'hidden' : 'visible',
      }"
    >
      <v-list-subheader>Conversations</v-list-subheader>
      <v-list-item
        v-for="(conv, i) in props.conversations || []"
        :key="i"
        :title="conv"
        prepend-icon="mdi-message-text"
      />
    </v-list>

    <!-- Footer with theme switch (only if not rail) -->
    <template #append>
      <div v-if="!rail">
        <v-divider />
        <div class="d-flex align-center justify-space-between px-3 py-2">
          <span>Theme</span>
          <v-switch
            v-model="isDark"
            hide-details
            inset
            true-icon="mdi-weather-night"
            false-icon="mdi-white-balance-sunny"
          />
        </div>
      </div>
    </template>
  </v-navigation-drawer>
</template>
