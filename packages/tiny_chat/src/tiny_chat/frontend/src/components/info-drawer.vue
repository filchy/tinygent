<script setup lang="ts">
import { emitter } from '@/services/event-bus'

const localDrawer = ref<boolean>(false)
const localSources = ref<SourceMessage[]>([])

onMounted(() => {
  emitter.on('displaySources', (sources: SourceMessage[] ) => {
    localDrawer.value = true
    localSources.value = sources
  })
})

const closeDrawer = () => {
  localDrawer.value = false

  nextTick(() => {
    sources.value = []
  })
}

const openUrl = (url: string) => {
    globalThis.open(url, '_blank')
}

const sourceTitle = (url: string) => {
  try {
    const parsed = new URL(url)
    let host = parsed.hostname || this.attributes.url
    if (host.startsWith('www.')) host = host.slice(4)
    return host
  } catch {
    return url
  }
}
</script>

<template>
  <v-navigation-drawer
    v-model='localDrawer'
    app
    location='right'
  >
    <template v-slot:prepend>
      <v-list-item>
        <template v-slot:prepend>
          <span class='font-weight-medium'>Sources</span>
        </template>
        <template v-slot:append>
          <v-icon
            icon='mdi-close'
            size='x-small'
            @click='localDrawer = false'
          />
        </template>
      </v-list-item>
    </template>

    <v-divider />

    <div class='d-flex flex-column px-2 pt-2' style='width: 100%;'>
      <v-card
        v-for='source in localSources'
        :key='source.id'
        class='d-flex flex-column pa-2'
        hover
        variant='text'
        @click="openUrl(source.url)"
      >
        <div class='text-caption d-flex align-center justify-end' style='gap: 4px;'>
          {{ sourceTitle(source.url) }}
          <div style='width: 16px;'>
            <v-img
              :src='source.favicon'
            ></v-img>
          </div>
        </div>
        <div class='text-caption font-weight-bold'>
          {{ source.name }}
        </div>
      </v-card>
    </div>
  </v-navigation-drawer>
</template>
