import MarkdownItAsync from 'markdown-it-async'
import { fromAsyncCodeToHtml } from '@shikijs/markdown-it/async'
import { codeToHtml } from 'shiki'

export async function createMarkdownRenderer(
  theme: 'light' | 'dark',
): Promise<ReturnType<typeof MarkdownItAsync>> {
  const md = MarkdownItAsync({
    html: true,
    linkify: true,
    typographer: true,
  })

  md.use(
    fromAsyncCodeToHtml(codeToHtml, {
      themes: {
        light: 'vitesse-light',
        dark: 'vitesse-dark',
      },
      defaultColor: theme,
    }),
  )

  return md
}
