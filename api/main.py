from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import os
import datetime

from api.routers import dataset_router, agent_router, model_router, project_router, deployment_router

API_PREFIX = "/api/v1"

def create_app() -> FastAPI:
    app = FastAPI(title="AutoDS API", version="1.0")

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(dataset_router.router, prefix=API_PREFIX)
    app.include_router(agent_router.router, prefix=API_PREFIX)
    app.include_router(model_router.router, prefix=API_PREFIX)
    app.include_router(project_router.router, prefix=API_PREFIX)
    app.include_router(deployment_router.router, prefix=API_PREFIX)

    # Exception handlers
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        return JSONResponse(status_code=422, content={"detail": exc.errors(), "body": exc.body})

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        return JSONResponse(status_code=500, content={"detail": str(exc)})

    @app.get("/")
    async def root():
        return {"service": "AutoDS API", "status": "running", "version": "1.0"}

    @app.get("/health")
    async def health():
        return {"status": "healthy", "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()}

    @app.on_event("startup")
    async def on_startup():
        # ensure storage dir
        os.makedirs("api", exist_ok=True)
        for fname in ("datasets.json", "api_jobs.json", "projects.json", "deployments.json"):
            path = os.path.join("api", fname)
            if not os.path.exists(path):
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write("{}" if fname.endswith("json") else "")

    @app.on_event("shutdown")
    async def on_shutdown():
        pass

    return app


app = create_app()
