import asyncio
from contextlib import contextmanager
import contextvars
import logging
import uuid

from fastapi import APIRouter
from fastapi import WebSocket
from fastapi import WebSocketDisconnect

from tiny_chat.context import current_chat_id
from tiny_chat.context import current_session
from tiny_chat.message import MessageUnion
from tiny_chat.message import UserMessage
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
            logger.info('Cancelled active task for session %s', session.session_id)


@contextmanager
def _run_in_context():
    ctx = contextvars.copy_context()
    try:
        yield ctx
    finally:
        pass


def _handle_user_message(session: WebsocketSession, msg: UserMessage):
    with _run_in_context() as ctx:
        current_chat_id.set(msg.chat_id)
        current_session.set(session)

        async def _run_message():
            try:
                result = await call_message(msg)

                logger.debug(
                    'Processed message on WebSocket session %s: %s',
                    session.session_id,
                    result,
                )
            except asyncio.CancelledError:
                logger.info(
                    'Message processing cancelled for session %s', session.session_id
                )
            except Exception as e:
                logger.exception(
                    'Error processing message on WebSocket session %s: %s',
                    session.session_id,
                    e,
                )

        session.active_task = asyncio.create_task(ctx.run(_run_message))


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

            msg = MessageUnion.validate_python(data)
            logger.debug(
                'Validated message on WebSocket session %s: %s', session_id, msg
            )

            if session.active_task and not session.active_task.done():
                session.active_task.cancel()

            if isinstance(msg, UserMessage):
                _handle_user_message(session, msg)
            else:
                logger.warning(
                    'Unsupported message type on WebSocket session %s: %s',
                    session_id,
                    msg.type,
                )

    except WebSocketDisconnect:
        logger.info('WebSocket disconnected for session %s', session_id)
        if session.active_task and not session.active_task.done():
            session.active_task.cancel()
    except Exception as e:
        logger.exception('Unexpected WebSocket error for session %s: %s', session_id, e)
