#!/usr/bin/env bash
#
# Nightly backup of the engine database to GCS.
#
# This exists because the database moved off Neon (which took its own
# backups) onto this VM on 2026-08-31. Nothing else backs it up.
#
# pg_dump runs INSIDE the container so the client version always matches the
# server; a host-side pg_dump silently breaks the day the image is bumped.
set -euo pipefail

BUCKET="gs://klio-engine-db-backups"
CONTAINER="klio-engine-engine-db-1"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
NAME="engine-db-${STAMP}.dump"
TMP="/tmp/${NAME}"

log() { echo "[$(date -u +%FT%TZ)] $*"; }

log "dumping ${CONTAINER} -> ${NAME}"
docker exec "$CONTAINER" pg_dump \
    --format=custom --no-owner --no-privileges \
    --file="/tmp/${NAME}" \
    "postgresql://klio@localhost/neondb"

docker cp "${CONTAINER}:/tmp/${NAME}" "$TMP"
# Keep a copy in the container purely so the verification below can read it
# with a matching-version pg_restore, then remove it.
docker exec "$CONTAINER" mv "/tmp/${NAME}" "/tmp/${NAME}.verify"

# A dump that cannot be listed is not a backup. Catch corruption here, while
# there is still a good copy on disk, rather than during a restore.
#
# This runs INSIDE the container: the VM host has no postgres client tools at
# all, so a host-side `pg_restore --list` fails for every dump, good or bad,
# and this guard would reject every backup forever. (It did exactly that on
# the first run, which is how the guard earned its keep.)
if ! docker exec "$CONTAINER" pg_restore --list "/tmp/${NAME}.verify" >/dev/null 2>&1; then
    log "FATAL: dump failed its own pg_restore --list check; not uploading"
    docker exec "$CONTAINER" rm -f "/tmp/${NAME}.verify" || true
    rm -f "$TMP"
    exit 1
fi
ENTRIES=$(docker exec "$CONTAINER" pg_restore --list "/tmp/${NAME}.verify" | grep -c '^[0-9]' || true)
docker exec "$CONTAINER" rm -f "/tmp/${NAME}.verify"
SIZE=$(stat -c%s "$TMP")
log "dump ok: ${SIZE} bytes, ${ENTRIES} TOC entries"

# A dump that suddenly shrinks by more than half is far more likely to be a
# broken backup than a real deletion; refuse it rather than let it rotate a
# good copy out of the 30-day window.
LAST_SIZE=$(cat /var/lib/klio-backup/last-size 2>/dev/null || echo 0)
if [ "$LAST_SIZE" -gt 0 ] && [ "$SIZE" -lt $((LAST_SIZE / 2)) ]; then
    log "FATAL: dump is ${SIZE}, less than half of the previous ${LAST_SIZE}; not uploading"
    rm -f "$TMP"
    exit 1
fi

gcloud storage cp "$TMP" "${BUCKET}/${NAME}" --quiet
mkdir -p /var/lib/klio-backup && echo "$SIZE" > /var/lib/klio-backup/last-size
rm -f "$TMP"
log "uploaded ${BUCKET}/${NAME}"

# Retention is enforced by the bucket's 30-day lifecycle rule, not here.
