import os
import logging
import redis

logger = logging.getLogger(__name__)

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)


def verify_redis_connection():
    try:
        redis_client.ping()
        logger.info("redis_stage = connection_verified")
    except Exception as error:
        logger.error(f"redis_stage = connection_failed error: {str(error)}")
        raise
