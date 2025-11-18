<script setup lang="ts">
import { ref, watch, computed } from 'vue'
import { useTheme } from 'vuetify'
import { createMarkdownRenderer } from '@/libs/markdown'

const props = defineProps<{ content: string }>()

const vuetifyTheme = useTheme()

const currentTheme = computed<'light' | 'dark'>(() =>
  vuetifyTheme.global.current.value.dark ? 'dark' : 'light',
)

const html = ref('')
let md = null

async function render() {
  md = await createMarkdownRenderer(currentTheme.value)
  html.value = await md.renderAsync(props.content || '')
}

render()

watch(currentTheme, render)
watch(() => props.content, render)
</script>

<template>
  <div class="markdown-body" v-html="html"></div>
</template>
