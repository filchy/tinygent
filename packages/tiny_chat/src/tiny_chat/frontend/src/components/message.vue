<script setup lang="ts">
import BaseMessage from './messages/base.vue'

const props = defineProps<{ msg: BaseMessage }>()

const componentMap: Record<string, any> = {
  text: defineAsyncComponent(() => import('./messages/text-message.vue')),
}
</script>

<template>
  <BaseMessage :msg="props.msg">
    <component
      :is="componentMap[props.msg.type]"
      :msg="props.msg"
      :color="props.msg.sender === 'user' ? 'primary' : 'grey-lighten-3'"
    />
  </BaseMessage>
</template>
