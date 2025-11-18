import { globalIgnores } from 'eslint/config'
import { defineConfigWithVueTs, vueTsConfigs } from '@vue/eslint-config-typescript'
import pluginVue from 'eslint-plugin-vue'
import skipFormatting from '@vue/eslint-config-prettier/skip-formatting'
import autoImports from './.eslintrc-auto-import.json' assert { type: 'json' }
import type { Linter } from 'eslint'

const autoImportGlobals: Linter.Config = {
  languageOptions: {
    globals: Object.fromEntries(
      Object.keys(autoImports.globals || {}).map((key) => [key, 'readonly' as const])
    )
  }
}

export default defineConfigWithVueTs(
  {
    name: 'app/files-to-lint',
    files: ['**/*.{ts,mts,tsx,vue}']
  },

  globalIgnores(['**/dist/**', '**/dist-ssr/**', '**/coverage/**']),

  pluginVue.configs['flat/essential'],
  vueTsConfigs.recommended,
  skipFormatting,
  autoImportGlobals,

  {
    rules: {
      'vue/multi-word-component-names': 'off',
      '@typescript-eslint/no-explicit-any': 'off'
    }
  }
)
