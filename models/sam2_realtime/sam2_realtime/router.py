import time
from fastapi import APIRouter, Depends, HTTPException, Request, status, Depends

from nimbro_vision_server.router_base import authenticate, get_manager_dep, logger
from nimbro_vision_server.model_base import BaseModel

# Every model subpackage should expose a router, even if it's empty.
# This router is included by ServerBase but currently has no additional endpoints.
router = APIRouter()

@router.post(
    "/reset",
    dependencies=[Depends(authenticate)]
)
async def unload_model(request: Request, mgr: BaseModel = Depends(get_manager_dep)):
    mgr.reset()
    logger.info("tracker reset")
    return {"reset": True}

@router.post(
    "/update",
    dependencies=[Depends(authenticate)]
)
async def update(request: Request, mgr: BaseModel = Depends(get_manager_dep)):
    """
    Initialize and (as soon as its implemented) update tracks.
    """
    try:
        payload = await request.json()
        logger.debug(f"Payload for update: {payload}")

        t0 = time.time()
        pre = mgr.preprocess(payload, mode="update")
        logger.debug(f"Preprocessing output: {pre}")

        t1 = time.time()
        raw = mgr.update(pre)
        logger.debug(f"Raw update output: {raw}")

        t2 = time.time()
        out = mgr.postprocess(raw)
        t3 = time.time()
        logger.debug(f"Postprocessing result: {out}")

        logger.info(
            "Timing breakdown (s)",
            extra={
                "phase": "preprocess",   "duration_s": t1 - t0,
                "phase": "update",       "duration_s": t2 - t1,
                "phase": "postprocess",  "duration_s": t3 - t2
            }
        )

    except HTTPException:
        # Propagate HTTP errors
        raise
    except Exception as e:
        logger.exception("Error during update pipeline")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Update failed: {e}"
        )

    logger.info("Update completed successfully")
    return out