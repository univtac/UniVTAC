"""FastAPI inference server for UniT deploy-time evaluation."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import traceback
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Request
import uvicorn


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _setup_paths() -> None:
    root = _repo_root()
    unit_dir = Path(__file__).resolve().parent
    os.chdir(root)
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(unit_dir))
    sys.path.insert(0, str(unit_dir / "third_party" / "diffusion_policy"))


def create_app(authkey: str) -> FastAPI:
    _setup_paths()
    from policy.UniT.deploy_policy import (
        _from_wire_payload,
        _to_wire_payload,
        _UniTLocalPolicy,
    )

    app = FastAPI(title="UniVTAC UniT Inference Server")
    app.state.authkey = authkey
    app.state.model = None

    def check_auth(header_auth: str | None, body_auth: str | None = None) -> None:
        expected = app.state.authkey
        if expected and header_auth != expected and body_auth != expected:
            raise HTTPException(status_code=401, detail="authentication failed")

    @app.get("/health")
    def health(x_unit_auth: str | None = Header(default=None)):
        check_auth(x_unit_auth)
        return {
            "status": "ok",
            "model_loaded": app.state.model is not None,
        }

    @app.post("/init")
    async def init(request: Request, x_unit_auth: str | None = Header(default=None)):
        payload = await request.json()
        check_auth(x_unit_auth, payload.get("authkey"))
        if app.state.model is None:
            app.state.model = _UniTLocalPolicy(_from_wire_payload(payload["args"]))
        return {"type": "ok"}

    @app.post("/act")
    async def act(request: Request, x_unit_auth: str | None = Header(default=None)):
        payload = await request.json()
        check_auth(x_unit_auth, payload.get("authkey"))
        if app.state.model is None:
            raise HTTPException(status_code=409, detail="model is not initialized")
        try:
            observation = _from_wire_payload(payload["observation"])
            action = app.state.model.get_action(observation)
            return {"type": "ok", "action": _to_wire_payload(action)}
        except BaseException as exc:
            return {
                "type": "error",
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }

    @app.post("/reset")
    async def reset(request: Request, x_unit_auth: str | None = Header(default=None)):
        payload = await request.json()
        check_auth(x_unit_auth, payload.get("authkey"))
        if app.state.model is not None:
            app.state.model.reset()
        return {"type": "ok"}

    @app.post("/shutdown")
    async def shutdown(request: Request, x_unit_auth: str | None = Header(default=None)):
        payload = await request.json()
        check_auth(x_unit_auth, payload.get("authkey"))
        asyncio.create_task(_shutdown_soon())
        return {"type": "ok"}

    async def _shutdown_soon() -> None:
        await asyncio.sleep(0.1)
        server = getattr(app.state, "server", None)
        if server is not None:
            server.should_exit = True

    return app


def main() -> int:
    parser = argparse.ArgumentParser(description="UniT FastAPI inference server")
    parser.add_argument("--host", default=os.environ.get("UNIT_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=os.environ.get("UNIT_PORT"))
    parser.add_argument("--authkey", default=os.environ.get("UNIT_AUTHKEY", ""))
    args = parser.parse_args()

    if args.port is None:
        raise ValueError("UNIT_PORT or --port is required")

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
