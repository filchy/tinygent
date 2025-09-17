import json
from typing import Any

from google.protobuf.struct_pb2 import Struct
import grpc
from szn.asistent.protobuf import asistentapi_pb2_grpc
from szn.asistent.protobuf.asistentapi_pb2 import ToolRequest
from szn.asistent.protobuf.asistentapi_pb2 import ToolResponse


class ToolHubService:
    _channel: grpc.aio.Channel | None = None
    _stub: asistentapi_pb2_grpc.ToolServiceStub | None = None

    @classmethod
    def _get_stub(cls) -> asistentapi_pb2_grpc.ToolServiceStub:
        if cls._channel is None or cls._stub is None:
            cls._channel = grpc.aio.insecure_channel(
                'asistent-toolhub.asistent.ftxt.dszn.cz:80'
            )
            cls._stub = asistentapi_pb2_grpc.ToolServiceStub(cls._channel)
        return cls._stub

    @classmethod
    async def call_tool(cls, name: str, args: dict) -> Any:
        stub_args = Struct()
        stub_args.update(args)

        response: ToolResponse = await cls._get_stub().Tool(
            ToolRequest(name=name, args=stub_args),
            timeout=20.0,
        )
        return response.response

    @classmethod
    async def call_tool_json(cls, name: str, args: dict) -> dict:
        response = await cls.call_tool(name, args)
        return json.loads(response)
