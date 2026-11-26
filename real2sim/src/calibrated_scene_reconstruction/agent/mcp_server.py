"""Expose all workflow JSON tools over MCP stdio using the installed optional transport."""
import argparse
import asyncio
import contextlib
import json
import sys

from mcp import types
from mcp.server import Server
from mcp.server.stdio import stdio_server

from .tool import DESCRIPTIONS, execute, schemas

_lock = asyncio.Lock()


async def list_tools(ctx, params):
    return types.ListToolsResult(tools=[types.Tool(name=name, description=DESCRIPTIONS[name],
        inputSchema={**schema['input'], 'properties': {**schema['input']['properties'], 'output': {'type': 'string'}},
                     'required': [*schema['input']['required'], 'output']}, outputSchema=schema['output'])
        for name, schema in schemas().items() if name in DESCRIPTIONS])


async def call_tool(ctx, params):
    try:
        args = dict(params.arguments or {})
        output = args.pop('output')
        if params.name == 'physics_status':
            result = await asyncio.to_thread(execute, params.name, args, output)
        else:
            async with _lock:
                with contextlib.redirect_stdout(sys.stderr):
                    result = await asyncio.to_thread(execute, params.name, args, output)
        return types.CallToolResult(content=[types.TextContent(type='text', text=json.dumps(result))],
                                   structuredContent=result, isError=not result['ok'])
    except Exception as exc:
        return types.CallToolResult(content=[types.TextContent(type='text', text=str(exc))], isError=True)


async def main():
    server = Server('calibrated_scene_reconstruction-workflow', on_list_tools=list_tools, on_call_tool=call_tool)
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    asyncio.run(main())
