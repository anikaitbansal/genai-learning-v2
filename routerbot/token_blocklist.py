import logging
from redis_client import redis_client

logger = logging.getLogger(__name__)

BLOCKLIST_PREFIX = "blocklist:jti:"


def add_to_blocklist(jti: str, expires_in_seconds: int) -> None:
    if expires_in_seconds <= 0:
        logger.info(
            f"blocklist_stage = add_skipped reason = token_already_expired jti: {jti}"
        )
        return

    key = f"{BLOCKLIST_PREFIX}{jti}"
    redis_client.set(key, "1", ex=expires_in_seconds)

    logger.info(
        f"blocklist_stage = add_done jti: {jti} ttl_seconds: {expires_in_seconds}"
    )


def is_blocklisted(jti: str) -> bool:
    key = f"{BLOCKLIST_PREFIX}{jti}"
    exists = redis_client.exists(key)

    is_present = exists == 1

    logger.info(f"blocklist_stage = check_done jti: {jti} is_blocklisted: {is_present}")
    return is_present
