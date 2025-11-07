import asyncio
import logging
import uuid

from fastapi import APIRouter
from fastapi import WebSocket
from fastapi import WebSocketDisconnect

from tiny_chat.message import BaseMessage
from tiny_chat.runtime import call_message
from tiny_chat.session import WebsocketSession

logger = logging.getLogger(__name__)

router = APIRouter()


def _handle_event(session: WebsocketSession, data: dict):
    event = data.get('event')
    task = session.active_task

    if event == 'stop':
        if task and not task.done():
            task.cancel()
            logger.info('Cancelled active task for session %s', session.id)


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

            if data.get('event'):
                _handle_event(session, data)
                continue

            msg = BaseMessage.model_validate(data)
            logger.debug(
                'Validated message on WebSocket session %s: %s', session_id, msg
            )
            session.history.append(msg)

            if session.active_task and not session.active_task.done():
                session.active_task.cancel()

            async def _run_message():
                try:
                    result = await call_message(msg)
                    logger.debug(
                        'Processed message on WebSocket session %s: %s',
                        session_id,
                        result,
                    )
                except asyncio.CancelledError:
                    logger.info(
                        'Message processing cancelled for session %s', session_id
                    )
                except Exception as e:
                    logger.exception(
                        'Error processing message on WebSocket session %s: %s',
                        session_id,
                        e,
                    )

            session.active_task = asyncio.create_task(_run_message())
    except WebSocketDisconnect:
        logger.info('WebSocket disconnected for session %s', session_id)
        if session.active_task and not session.active_task.done():
            session.active_task.cancel()
    except Exception as e:
        logger.exception('Unexpected WebSocket error for session %s: %s', session_id, e)
