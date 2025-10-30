import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from tiny_chat.ws import router as ws_router
from tiny_chat.http import router as http_router
from tiny_chat.config import PACKAGE_ROOT


def _get_build_dir(local_target: str, packaged_target: str) -> str:
    local_build_dir = os.path.join(PACKAGE_ROOT, local_target, 'dist')
    packaged_build_dir = os.path.join(PACKAGE_ROOT, packaged_target, 'dist')

    if os.path.exists(local_build_dir):
        return local_build_dir
    elif os.path.exists(packaged_build_dir):
        return packaged_build_dir
    else:
        raise FileNotFoundError(
            f'No frontend build found in {local_build_dir} or {packaged_build_dir}'
        )


build_dir = _get_build_dir('tiny_chat/frontend', 'tiny_chat/frontend')

app = FastAPI()
app.include_router(ws_router)
app.include_router(http_router)

if os.path.exists(build_dir):
    app.mount('/', StaticFiles(directory=build_dir, html=True), name='static')


def run(host: str = '127.0.0.1', port: int = 8000, reload: bool = False):
    import uvicorn

    uvicorn.run('tiny_chat.server:app', host=host, port=port, reload=reload)
