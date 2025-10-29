<script setup lang='ts'>
import lightAvatar from '@/assets/light-avatar.png'
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
    <v-list class='pa-0'>
      <v-list-item>
        <template #prepend>
          <v-img
            :src="darkAvatar"
            width="28"
            height="64"
            class="mr-2"
            cover
          />
        </template>

        <template #append>
          <v-btn
            icon
            variant='text'
            @click.stop='rail = !rail'
          >
            <v-icon>{{ rail ? 'mdi-chevron-right' : 'mdi-chevron-left' }}</v-icon>
          </v-btn>
        </template>
      </v-list-item>
    </v-list>

    <v-divider></v-divider>

    <v-list nav dense>
      <v-list-subheader v-show='!rail'>Conversations</v-list-subheader>
      <v-list-item
        v-for='(conv, i) in props.conversations || []'
        :key='i'
        :title='!rail ? conv : ""'
        prepend-icon='mdi-message-text'
      />
    </v-list>
  </v-navigation-drawer>
</template>
