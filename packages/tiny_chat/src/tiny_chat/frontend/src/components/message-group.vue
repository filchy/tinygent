<script setup lang="ts">
import { useTheme } from 'vuetify'

import { isSourceMessage, isToolCallMessage } from '@/utils/message-utils'
import { emitter } from '@/services/event-bus'
import darkAvatar from '@/assets/dark-avatar.png'
import lightAvatar from '@/assets/light-avatar.png'

import MarkdownRenderer from './markdown-renderer.vue'

const props = defineProps<{ messageGroup: MessageGroup }>()

const isOpenedToolCalls = ref<boolean>(false)

const theme = useTheme()

const currentAvatar = computed(() => (theme.global.current.value.dark ? lightAvatar : darkAvatar))

const main = props.messageGroup.main
const children = props.messageGroup.children ?? []

const isLoading = computed(() => main?.type === 'loading')

const toolCalls = computed(() => children.filter((c) => isToolCallMessage(c)))
const sources = computed(() => children.filter((c) => isSourceMessage(c)))

const openInfoDrawer = () => {
  emitter.emit('displaySources', { sources: sources.value })
}
</script>

<template>
  <v-card variant="flat" density="default" style="background: transparent !important">
    <div class="d-flex flex-column">
      <div class="d-flex flex-row">
        <div class="d-flex flex-column justify-start" style="width: 36px; user-select: none">
          <v-avatar size="36" class="mr-2 flex-shrink-0">
            <v-img :src="currentAvatar" />
          </v-avatar>
        </div>

        <div class="d-flex flex-column" style="width: 100%">
          <div
            v-if="toolCalls.length > 0"
            class="d-flex align-center text-body-1 text-grey-darken-1"
            style="height: 36px; cursor: pointer; user-select: none"
            @click.stop="isOpenedToolCalls = !isOpenedToolCalls"
          >
            {{ toolCalls.length }} tool call{{ toolCalls.length > 1 ? 's' : '' }} made

            <v-spacer />

            <v-icon
              :icon="isOpenedToolCalls ? 'mdi-chevron-up' : 'mdi-chevron-down'"
              class="mr-2"
              style="cursor: pointer"
            />
          </div>

          <v-expand-transition>
            <div
              v-if="isOpenedToolCalls"
              class="d-flex flex-column position-relative"
              style="margin-left: -36px"
            >
              <div
                v-for="(toolCall, i) in toolCalls"
                :key="i"
                class="d-flex align-center text-grey-darken-1"
                style="width: 100%; height: 36px"
              >
                <div
                  class="d-flex align-center justify-center"
                  style="width: 36px; position: relative"
                >
                  <v-icon
                    icon="mdi-web"
                    :color="theme.global.current.value.dark ? 'info-lighten-1' : 'info-darken-1'"
                    size="14"
                  />
                </div>

                <div
                  class="pl-2 text-body-2 text-grey-darken-1 d-flex align-center"
                  style="width: 100%"
                >
                  <span class="font-weight-bold">{{ toolCall.tool_name }}</span>
                  <v-spacer />
                  <v-tooltip location="top" max-width="400">
                    <template #activator="{ props }">
                      <span v-bind="props">
                        {{
                          ((s) => (s.length > 30 ? s.slice(0, 30) + '…' : s))(
                            JSON.stringify(toolCall.tool_args),
                          )
                        }}
                      </span>
                    </template>

                    {{ JSON.stringify(toolCall.tool_args, null, 2) }}
                  </v-tooltip>
                </div>
              </div>
            </div>
          </v-expand-transition>

          <div
            style="min-height: 36px"
            class="d-flex align-center text-body-1 py-3"
            v-if="!isLoading"
          >
            <markdown-renderer :content="main?.content ?? ''" />
          </div>

          <div class="d-flex align-center" style="height: 36px" v-else>
            <div class="loading-wrapper">
              <span class="loading-text">thinking</span>
              <span class="wave-dots"> <span>.</span><span>.</span><span>.</span> </span>
            </div>
          </div>

          <div class="d-flex align-center justify-start" v-if="sources.length > 0">
            <v-chip label variant="outlined" @click.stop="openInfoDrawer">
              {{ sources.length <= 3 ? sources.length : '3+' }} source{{
                sources.length > 1 ? 's' : ''
              }}
            </v-chip>
          </div>
        </div>
      </div>
    </div>
  </v-card>
</template>

<style scoped>
.loading-wrapper {
  display: inline-flex;
  align-items: center;
  gap: 4px;
}

.loading-text {
  font-size: 1.1rem;
  font-weight: 500;
  background: linear-gradient(90deg, #d0d0d0, #a0a0a0, #d0d0d0);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  animation: shimmer 2s infinite linear;
}

.wave-dots {
  display: inline-flex;
  gap: 4px;
  font-size: 1.4rem;
  color: silver;
}

.wave-dots span {
  display: inline-block;
  animation: wave 1.2s infinite ease-in-out;
}

.wave-dots span:nth-child(2) {
  animation-delay: 0.2s;
}

.wave-dots span:nth-child(3) {
  animation-delay: 0.4s;
}

@keyframes wave {
  0%,
  60%,
  100% {
    transform: translateY(0);
    opacity: 0.6;
  }
  30% {
    transform: translateY(-6px);
    opacity: 1;
  }
}

@keyframes shimmer {
  0% {
    background-position: -100px 0;
  }
  100% {
    background-position: 100px 0;
  }
}
</style>
