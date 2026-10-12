import os
from pathlib import Path

TEST_DATA = Path(__file__).parent / ".test-data"
os.environ["DATA_DIR"] = str(TEST_DATA)
os.environ["DATABASE_URL"] = f"sqlite:///{(TEST_DATA / 'test.db').as_posix()}"

import pytest
import httpx

from app.database import Base, engine
from app.main import app


@pytest.fixture(autouse=True)
def clean_database():
    TEST_DATA.mkdir(parents=True, exist_ok=True)
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def client():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as test_client:
        yield test_client
