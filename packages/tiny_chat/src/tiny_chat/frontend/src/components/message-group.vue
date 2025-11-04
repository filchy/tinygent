<script setup lang="ts">
import { useTheme } from 'vuetify'

import darkAvatar from '@/assets/dark-avatar.png'
import lightAvatar from '@/assets/light-avatar.png'

const props = defineProps<{ messageGroup: MessageGroup }>()

const isOpenedToolCalls = ref<boolean>(false)

const theme = useTheme()
const infoColor = computed(() => theme.global.current.value.colors.info)

const currentAvatar = computed(() =>
  theme.global.current.value.dark ? lightAvatar : darkAvatar
)

const main = props.messageGroup.main
const children = props.messageGroup.children

const toolCalls = computed(() => children.filter(c => c.type === 'tool'))
</script>

<template>
  <v-card variant='flat' density='default'>
    <div class='d-flex flex-column'>
      <div class='d-flex flex-row'>
        <div
          class='d-flex flex-column justify-start'
          style='width: 36px;'
        >
          <v-avatar size='36' class='mr-2 flex-shrink-0'>
            <v-img :src="currentAvatar" />
          </v-avatar>
        </div>

        <div
          class='d-flex flex-column'
          style='width: 100%;'
        >
          <div
            v-if='toolCalls.length > 0'
            class='d-flex align-center text-body-1 text-grey-darken-1'
            style='height: 36px; cursor: pointer;'
            @click.stop='isOpenedToolCalls = !isOpenedToolCalls'
          >
            {{ toolCalls.length }} tool call{{ toolCalls.length > 1 ? 's' : '' }} made

            <v-spacer />

            <v-icon
              :icon='isOpenedToolCalls ? "mdi-chevron-up" : "mdi-chevron-down"'
              class='mr-2'
              style='cursor: pointer;'
            />
          </div>

          <div
            v-if='isOpenedToolCalls'
            class='d-flex flex-column position-relative'
            style='margin-left:-36px;'
          >
            <div
              v-for='(toolCall, i) in toolCalls'
              :key='i'
              class='d-flex align-center text-grey-darken-1'
              style='width: 100%; height: 36px;'
            >
              <div
                class='d-flex align-center justify-center'
                style='
                  width: 36px;
                  position: relative;
                '
              >
                <v-icon
                  icon='mdi-web'
                  :color='theme.global.current.value.dark ? "info-lighten-1" : "info-darken-1"'
                  size='14'
                />
              </div>

              <div class='pl-2 text-body-2 text-grey-darken-1 d-flex align-center' style='width: 100%;'>
                <span class='font-weight-bold'>{{ toolCall.tool_name }}</span>
                <v-spacer />
                <v-tooltip location='top' max-width='400'>
                  <template #activator='{ props }'>
                    <span v-bind='props'>
                      {{
                        (s => s.length > 30 ? s.slice(0, 30) + '…' : s)
                        (JSON.stringify(toolCall.tool_args))
                      }}
                    </span>
                  </template>

                  {{ JSON.stringify(toolCall.tool_args, null, 2) }}
                </v-tooltip>
              </div>
            </div>
          </div>

          <div style='min-height: 36px;' class='d-flex align-center text-body-1'>
            {{ main?.content }}
          </div>
        </div>
      </div>
    </div>
  </v-card>
</template>
