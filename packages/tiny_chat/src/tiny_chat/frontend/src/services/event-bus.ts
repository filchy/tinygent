import mitt from 'mitt'

type Events = {
  displaySources: { sources: SourceMessage[] }
}

export const emitter = mitt<Events>()
