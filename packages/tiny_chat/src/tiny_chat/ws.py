import logging
import uuid
from fastapi import APIRouter
from fastapi import WebSocketDisconnect
from fastapi import WebSocket

from tiny_chat.message import BaseMessage
from tiny_chat.runtime import call_message
from tiny_chat.session import WebsocketSession

logger = logging.getLogger(__name__)

router = APIRouter()


@router.websocket('/ws')
async def websocket_handler(ws: WebSocket):
    await ws.accept()

    socket_id = str(uuid.uuid4())
    session_id = str(uuid.uuid4())
    session = WebsocketSession(session_id, socket_id, ws)

    try:
        while True:
            data = await ws.receive_json()
            logger.debug('Received data on WebSocket session %s: %s', session_id, data)

            msg = BaseMessage.model_validate(data)
            logger.debug('Validated message on WebSocket session %s: %s', session_id, msg)
            session.history.append(msg)

            result = await call_message(msg)
            logger.debug('Processed message on WebSocket session %s: %s', session_id, result)
    except WebSocketDisconnect:
        logger.info('WebSocket disconnected for session %s', session_id)
    except Exception as e:
        logger.exception('Unexpected WebSocket error for session %s: %s', session_id, e)
