"""FastAPI inference server for SmolVLA deploy-time evaluation.

Run this file with ``policy/smolvla/.venv/bin/python``. The IsaacLab eval
process talks to it over HTTP so the simulator and SmolVLA dependency stacks
can stay in separate Python environments.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Request
import uvicorn


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _setup_paths() -> None:
    root = _repo_root()
    smolvla_dir = Path(__file__).resolve().parent
    os.chdir(root)
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(smolvla_dir / "src"))


def create_app(authkey: str) -> FastAPI:
    _setup_paths()
    from policy.smolvla.deploy_policy import (
        _from_wire_payload,
        _SmolVLALocalPolicy,
        _to_wire_payload,
    )

    app = FastAPI(title="UniVTAC SmolVLA Inference Server")
    app.state.authkey = authkey
    app.state.model = None
    app.state.shutdown_requested = False

    def check_auth(header_auth: str | None, body_auth: str | None = None) -> None:
        expected = app.state.authkey
        if expected and header_auth != expected and body_auth != expected:
            raise HTTPException(status_code=401, detail="authentication failed")

    @app.get("/health")
    def health(x_smolvla_auth: str | None = Header(default=None)):
        check_auth(x_smolvla_auth)
        return {
            "status": "ok",
            "model_loaded": app.state.model is not None,
        }

    @app.post("/init")
    async def init(request: Request, x_smolvla_auth: str | None = Header(default=None)):
        payload = await request.json()
        check_auth(x_smolvla_auth, payload.get("authkey"))
        if app.state.model is None:
            app.state.model = _SmolVLALocalPolicy(_from_wire_payload(payload["args"]))
        return {"type": "ok"}

    @app.post("/act")
    async def act(request: Request, x_smolvla_auth: str | None = Header(default=None)):
        payload = await request.json()
        check_auth(x_smolvla_auth, payload.get("authkey"))
        if app.state.model is None:
            raise HTTPException(status_code=409, detail="model is not initialized")
        try:
            observation = _from_wire_payload(payload["observation"])
            action = app.state.model.get_action(
                observation,
                payload.get("instruction"),
            )
            return {"type": "ok", "action": _to_wire_payload(action)}
        except BaseException as exc:
            return {
                "type": "error",
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }

    @app.post("/reset")
    async def reset(request: Request, x_smolvla_auth: str | None = Header(default=None)):
        payload = await request.json()
        check_auth(x_smolvla_auth, payload.get("authkey"))
        if app.state.model is not None:
            app.state.model.reset()
        return {"type": "ok"}

    @app.post("/shutdown")
    async def shutdown(request: Request, x_smolvla_auth: str | None = Header(default=None)):
        payload = await request.json()
        check_auth(x_smolvla_auth, payload.get("authkey"))
        app.state.shutdown_requested = True
        asyncio.create_task(_shutdown_soon())
        return {"type": "ok"}

    async def _shutdown_soon() -> None:
        await asyncio.sleep(0.1)
        server = getattr(app.state, "server", None)
        if server is not None:
            server.should_exit = True

    return app


def main() -> int:
    parser = argparse.ArgumentParser(description="SmolVLA FastAPI inference server")
    parser.add_argument("--host", default=os.environ.get("SMOLVLA_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=os.environ.get("SMOLVLA_PORT"))
    parser.add_argument("--authkey", default=os.environ.get("SMOLVLA_AUTHKEY", ""))
    args = parser.parse_args()

    if args.port is None:
        raise ValueError("SMOLVLA_PORT or --port is required")

    app = create_app(args.authkey)
    config = uvicorn.Config(
        app,
        host=args.host,
        port=int(args.port),
        log_level="info",
        access_log=False,
    )
    server = uvicorn.Server(config)
    app.state.server = server
    server.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
