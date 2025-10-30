import uuid
from fastapi import APIRouter
from fastapi import WebSocket

from tiny_chat.message import UserMessage
from tiny_chat.runtime import call_message
from tiny_chat.session import WebsocketSession


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
            msg = UserMessage.model_validate(data)
            session.history.append(msg)

            result = await call_message(msg)

            if result is not None:
                await session.emit(result)
    except Exception:
        await ws.close()
