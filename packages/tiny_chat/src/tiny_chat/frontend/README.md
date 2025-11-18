# tiny-chat frontend

This package ships a Vue 3 + Vite frontend that is published together with the Python `tiny-chat` API. The frontend code lives in `src/tiny_chat/frontend` and is bundled into `dist/` during builds.

## Prerequisites

- [nvm (Node Version Manager)](https://github.com/nvm-sh/nvm#installing-and-updating) – required to install and pin the correct Node.js runtime.
- Node.js 22 (the project targets `>=22.12.0 <23`).
- npm (bundled with Node.js).

```bash
# install nvm (see project docs for alternative install methods)
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash

# load nvm in your current shell
export NVM_DIR="$([ -z "${XDG_CONFIG_HOME-}" ] && printf %s "${HOME}/.nvm" || printf %s "${XDG_CONFIG_HOME}/nvm")"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"

# install and activate Node 22 for this project
nvm install 22
nvm use 22

# optional: make Node 22 the default
nvm alias default 22
```

> **Tip:** run `nvm use 22` in every new shell before working on the frontend to ensure the correct engine version.

## Install dependencies

From `packages/tiny_chat/src/tiny_chat/frontend`:

```bash
npm install
```

## Local workflows

- **Run the dev server:** `npm run dev`
- **Create a production build:** `npm run build`
- **Preview the production build:** `npm run preview`

The `build` script runs `vue-tsc` via `npm run type-check` before bundling with Vite. The compiled output lands in `src/tiny_chat/frontend/dist`.

## Quality tooling

- **Lint with ESLint + auto-fix:** `npm run lint`
- **Format with Prettier:** `npm run format`
- **Type-check with vue-tsc:** `npm run type-check`

Run these commands from `packages/tiny_chat/src/tiny_chat/frontend`.

## Python package integration note

When iterating locally on the Python package, remove any stale frontend artifacts so Hatch picks up the fresh sources:

```bash
rm -rf packages/tiny_chat/src/tiny_chat/frontend/dist
```

If `dist/` exists, Hatch will bundle those files automatically (see `[tool.hatch.build]` in `pyproject.toml`). Deleting the directory ensures the Python package serves your live `npm run dev` assets or newly built output.

## Next steps

1. Keep `nvm use 22` in your shell startup (e.g., by adding it to `.bashrc`) to avoid version drift.
2. Add the lint, format, and build commands to CI to enforce consistent quality.
