# nimbro_vision_server/router_base.py

import os
import json
import time
import torch
from fastapi import APIRouter, Depends, HTTPException, Request, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from .model_base import BaseModel
from .utils import setup_logging

logger = setup_logging()
common_router = APIRouter()

# ——— auth dependency ———

auth_scheme = HTTPBearer(auto_error=False)

async def authenticate(
    creds: HTTPAuthorizationCredentials = Depends(auth_scheme)
) -> str:
    if not creds or creds.scheme.lower() != "bearer":
        logger.warning("Missing or invalid auth scheme")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid auth scheme",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = creds.credentials
    expected = os.getenv("NIMBRO_VISION_SERVER_TOKEN")
    if expected and token != expected:
        logger.warning("Invalid token")
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                            detail="Invalid token")
    return token

# ——— helper to grab the live manager ———

def get_manager_dep(request: Request) -> BaseModel:
    mgr = request.app.state.manager
    if mgr is None:
        raise HTTPException(503, "No model loaded")
    return mgr

# ——— shared endpoints ———

@common_router.get("/health")
async def health():
    return {
        "status": "ok",
        "cuda_available": torch.cuda.is_available()
    }

@common_router.get(
    "/model_flavors",
    dependencies=[Depends(authenticate)]
)
async def model_flavors(request: Request):
    ModelClass = request.app.state.ModelClass
    flavors = ModelClass.get_available_flavors()
    logger.info(f"flavors available: {flavors}")
    return {"flavors": flavors}

@common_router.post(
    "/load",
    dependencies=[Depends(authenticate)]
)
async def load_model(request: Request):
    """
    Load (or reload) the model.  
    The JSON body is forwarded to BaseModel.load().
    If a model is already loaded, it will be unloaded first.
    """
    payload = await request.json()
    start = time.time()

    # if there’s already a model, unload it first
    current_mgr = request.app.state.manager
    if current_mgr is not None:
        try:
            current_mgr.unload()
            logger.info("Existing model instance unloaded")
        except Exception as e:
            logger.warning(f"Error unloading existing model: {e}")
        finally:
            request.app.state.manager = None

    # instantiate & load the new model
    ModelClass = request.app.state.ModelClass
    mgr = ModelClass()
    try:
        mgr.load(payload)   # payload can contain anything the model needs
        duration = time.time() - start
        logger.info(f"Model.load() took {duration:.3f}s", extra={"phase": "load", "duration_s": duration})
    except Exception as e:
        # if load fails, ensure we don't leave a half‐loaded manager
        logger.error(f"Failed to load model with payload {payload}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model load error: {e}"
        )

    request.app.state.manager = mgr
    logger.info(f"Loaded new model instance ({ModelClass.__name__}) with payload: {payload}")
    return {
        "loaded_model": ModelClass.__name__,
        **({"flavor": payload.get("flavor")} if "flavor" in payload else {})
    }

@common_router.post(
    "/unload",
    dependencies=[Depends(authenticate)]
)
async def unload_model(request: Request, mgr: BaseModel = Depends(get_manager_dep)):
    mgr.unload()
    request.app.state.manager = None
    logger.info("model unloaded")
    return {"unloaded": True}

@common_router.post(
    "/infer",
    dependencies=[Depends(authenticate)]
)
async def infer(request: Request, mgr: BaseModel = Depends(get_manager_dep)):
    try:
        payload = await request.json()
        logger.debug(f"Payload for inference: {payload}")

        t0 = time.time()
        pre = mgr.preprocess(payload)
        logger.debug(f"Preprocessing output: {pre}")

        t1 = time.time()
        raw = mgr.infer(pre)
        logger.debug(f"Inference raw output: {raw}")

        t2 = time.time()
        out = mgr.postprocess(raw)
        t3 = time.time()
        logger.debug(f"Postprocessing result: {out}")
        timings = {"preprocess:  duration_s": t1 - t0,
                   "infer:       duration_s": t2 - t1,
                   "postprocess: duration_s": t3 - t2}
        logger.info(f"Timing breakdown (s)\n{json.dumps(timings, indent=4)}")

    except HTTPException:
        # if any explicit HTTPException is raised, re-raise it
        raise

    except Exception as e:
        # everything else becomes a 500
        logger.exception("Error during inference pipeline")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {e}"
        )

    logger.info("Inference completed successfully")
    return out

@common_router.get(
    "/status",
    dependencies=[Depends(authenticate)]
)
async def model_flavors(request: Request):
    ModelClass = request.app.state.ModelClass
    model_family = ModelClass.get_name()
    current_mgr = request.app.state.manager
    if current_mgr is not None:
        status = current_mgr.get_status()
    else:
        status = None
    return {"model_family": model_family, "status": status}