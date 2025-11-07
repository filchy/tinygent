from fastapi import APIRouter

from tiny_chat.config import VERSION

router = APIRouter(prefix='/api')


@router.get('/health')
async def health_check():
    return {'status': 'ok'}


@router.get('/version')
async def version_check():
    return {'version': VERSION}
