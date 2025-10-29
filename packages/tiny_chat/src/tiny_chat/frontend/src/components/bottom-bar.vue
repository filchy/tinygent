<script setup lang='ts'>
import { useTheme } from 'vuetify'
import darkAvatar from '@/assets/dark-avatar.png'
import lightAvatar from '@/assets/light-avatar.png'

const theme = useTheme()

const currentAvatar = computed(() =>
  theme.global.current.value.dark ? lightAvatar : darkAvatar
)
</script>

<template>
  <v-footer
    app
    height='40'
    class='d-flex align-center justify-center text-caption'
  >
    <span class='d-flex align-center' style='font-size: 10px;'>
      Build with
      <v-img
        :src='currentAvatar'
        alt='tinygent logo'
        width='32'
        height='32'
        class='mx-1'
        contain
      />
      tinygent
    </span>
  </v-footer>
</template>
