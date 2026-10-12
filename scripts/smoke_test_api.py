"""Small no-network API smoke test using HTTPX's in-process ASGI transport."""

import asyncio

import httpx

from app.main import app


async def main() -> None:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v1/health")
        response.raise_for_status()
        payload = response.json()
        assert payload["status"] == "ok"
        print(payload)


if __name__ == "__main__":
    asyncio.run(main())
