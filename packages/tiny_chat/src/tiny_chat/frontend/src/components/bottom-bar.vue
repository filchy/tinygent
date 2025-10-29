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
    class='d-flex align-center justify-center text-caption font-weight-thin'
    color='transparent'
  >
    <span class='d-flex align-center text-caption'>
      Build with
      <v-img
        :src='currentAvatar'
        alt='tinygent logo'
        width='32'
        height='32'
        class='mx-1'
        contain
      />
      <a
        class='font-weight-bold'
        href='https://github.com/filchy/tinygent'
        style='text-decoration: none; color: inherit;'
      >
        tinygent
      </a>
    </span>
  </v-footer>
</template>
