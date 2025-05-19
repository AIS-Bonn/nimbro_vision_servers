# nimbro_vision_server/server_base.py

import importlib
import os
import time
import uvicorn
from fastapi import FastAPI, Request, HTTPException, status
from .router_base import common_router
from .utils import setup_logging

logger = setup_logging()

class ServerBase:
    def __init__(
        self,
        model_family: str,
        host: str = "0.0.0.0",
        port: int = 8000,
        token: str | None = None,
        preload_flavor: str | None = None,
    ):
        """
        A server for a single model family.

        Args:
            model_family: Name of the family folder under `models/` to load custom routes from.
            host/port: where to bind.
            token: optional Bearer token required for all protected routes.
        """
        self.app = FastAPI()
        self.host = host
        self.port = port
        self.token = token

        # init model - seperate class and instance for common router
        model_mod = importlib.import_module(f"models.{model_family}.{model_family}.model")
        ModelClass = getattr(model_mod, "Model")
        self.app.state.ModelClass = ModelClass
        self.app.state.manager = None

        if preload_flavor is not None:
            logger.info(f"Preloading model flavor '{preload_flavor}'")
            mgr = ModelClass()
            try:
                start = time.time()
                mgr.load({"flavor": preload_flavor})   # payload can contain anything the model needs
                duration = time.time() - start
                logger.info(f"Model.load() took {duration:.3f}s", extra={"phase": "load", "duration_s": duration})
            except Exception as e:
                # if load fails, ensure we don't leave a half‐loaded manager
                logger.error(f"Failed to load model with flavor '{preload_flavor}': {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Model load error: {e}"
                )
            self.app.state.manager = mgr

        # mount shared endpoints (/health, /model_flavors, /load, /unload, /infer)
        self.app.include_router(common_router)

        # mount custom endpoints for this model family
        try:
            mod = importlib.import_module(f"models.{model_family}.{model_family}.router")
            self.app.include_router(mod.router)
            logger.info(f"Mounted custom router for family '{model_family}'")
        except ModuleNotFoundError:
            logger.warning(f"No custom router found for family '{model_family}'")

        # middleware for request logging
        @self.app.middleware("http")
        async def log_requests(request: Request, call_next):
            logger.info(f"→ {request.method} {request.url.path}")
            resp = await call_next(request)
            logger.info(f"← {request.method} {request.url.path} {resp.status_code}")
            return resp
        
        @self.app.on_event("shutdown")
        async def shutdown_event():
            mgr = self.app.state.manager
            if mgr is not None:
                try:
                    mgr.unload()
                    logger.info("Model unloaded on shutdown")
                except Exception as e:
                    logger.warning(f"Error unloading model on shutdown: {e}")

    def run(self):
        # Pass the token into router_base via env so it can enforce auth
        logger.info("Starting server...")
        if self.token:
            os.environ["NIMBRO_VISION_SERVER_TOKEN"] = self.token
        logger.info(f"Token: {self.token}")
        uvicorn.run(self.app, host=self.host, port=self.port)
