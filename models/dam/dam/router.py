from fastapi import APIRouter

# Every model subpackage should expose a router, even if it's empty.
# This router is included by ServerBase but currently has no additional endpoints.
router = APIRouter()