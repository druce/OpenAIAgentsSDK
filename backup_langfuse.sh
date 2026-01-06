#!/bin/bash
#
# Langfuse PostgreSQL Database Backup Script
#
# This script backs up the Langfuse PostgreSQL database using TWO methods:
# 1. pg_dump (logical backup - portable SQL dump)
# 2. Docker volume backup (physical backup - complete volume snapshot)
#
# Usage: ./backup_langfuse.sh
# Cron:  0 2 * * * /Users/drucev/projects/OpenAIAgentsSDK/backup_langfuse.sh >> /Users/drucev/projects/OpenAIAgentsSDK/langfuse-backups/backup.log 2>&1
#

# Configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BACKUP_DIR="$SCRIPT_DIR/langfuse-backups"
VOLUME_BACKUP_DIR="$SCRIPT_DIR/langfuse-backups/volumes"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_FILE="langfuse_backup_${TIMESTAMP}.sql.gz"
VOLUME_BACKUP_FILE="langfuse_volume_${TIMESTAMP}.tar.gz"
RETENTION_COUNT=7
POSTGRES_USER="postgres"
POSTGRES_DB="postgres"
DOCKER_COMPOSE_DIR="/Users/drucev/projects/OpenAIAgentsSDK/langfuse"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

log_error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}" >&2
}

log_success() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] SUCCESS: $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

# Create backup directories if they don't exist
if [ ! -d "$BACKUP_DIR" ]; then
    log "Creating backup directory: $BACKUP_DIR"
    mkdir -p "$BACKUP_DIR"
    if [ $? -ne 0 ]; then
        log_error "Failed to create backup directory"
        exit 1
    fi
fi

if [ ! -d "$VOLUME_BACKUP_DIR" ]; then
    log "Creating volume backup directory: $VOLUME_BACKUP_DIR"
    mkdir -p "$VOLUME_BACKUP_DIR"
    if [ $? -ne 0 ]; then
        log_error "Failed to create volume backup directory"
        exit 1
    fi
fi

# Check if docker is running
if ! docker info > /dev/null 2>&1; then
    log_error "Docker is not running"
    exit 1
fi

# Find the postgres container
log "Looking for Langfuse postgres container..."
POSTGRES_CONTAINER=$(docker ps --filter "name=postgres" --format "{{.Names}}" | grep -i "langfuse" | head -1)

if [ -z "$POSTGRES_CONTAINER" ]; then
    log_error "Could not find Langfuse postgres container. Is it running?"
    log "Checking all running containers with 'postgres' in name:"
    docker ps --filter "name=postgres" --format "table {{.Names}}\t{{.Status}}"
    exit 1
fi

log "Found postgres container: $POSTGRES_CONTAINER"

# Check available disk space (require at least 1GB free)
AVAILABLE_SPACE=$(df -k "$BACKUP_DIR" | tail -1 | awk '{print $4}')
REQUIRED_SPACE=1048576  # 1GB in KB

if [ "$AVAILABLE_SPACE" -lt "$REQUIRED_SPACE" ]; then
    log_warning "Low disk space: $(($AVAILABLE_SPACE / 1024)) MB available"
fi

# Perform the backup
log "Starting backup of database '$POSTGRES_DB' from container '$POSTGRES_CONTAINER'..."
BACKUP_PATH="$BACKUP_DIR/$BACKUP_FILE"

docker exec "$POSTGRES_CONTAINER" pg_dump -U "$POSTGRES_USER" "$POSTGRES_DB" | gzip > "$BACKUP_PATH"

# Check if backup was successful
if [ $? -ne 0 ]; then
    log_error "Backup failed"
    # Clean up failed backup file if it exists
    [ -f "$BACKUP_PATH" ] && rm "$BACKUP_PATH"
    exit 1
fi

# Verify backup file exists and has content
if [ ! -f "$BACKUP_PATH" ]; then
    log_error "Backup file was not created"
    exit 1
fi

BACKUP_SIZE=$(stat -f%z "$BACKUP_PATH" 2>/dev/null || stat -c%s "$BACKUP_PATH" 2>/dev/null)
if [ "$BACKUP_SIZE" -eq 0 ]; then
    log_error "Backup file is empty"
    rm "$BACKUP_PATH"
    exit 1
fi

# Convert size to human-readable format
BACKUP_SIZE_MB=$(echo "scale=2; $BACKUP_SIZE / 1048576" | bc)
log_success "Backup created: $BACKUP_FILE (${BACKUP_SIZE_MB} MB)"

# Manage backup retention (keep last N backups)
log "Managing pg_dump backup retention (keeping last $RETENTION_COUNT backups)..."
BACKUP_COUNT=$(ls -1 "$BACKUP_DIR"/langfuse_backup_*.sql.gz 2>/dev/null | wc -l)

if [ "$BACKUP_COUNT" -gt "$RETENTION_COUNT" ]; then
    DELETE_COUNT=$((BACKUP_COUNT - RETENTION_COUNT))
    log "Found $BACKUP_COUNT pg_dump backups, deleting oldest $DELETE_COUNT..."

    ls -1t "$BACKUP_DIR"/langfuse_backup_*.sql.gz | tail -n "$DELETE_COUNT" | while read -r old_backup; do
        log "Deleting old backup: $(basename "$old_backup")"
        rm "$old_backup"
    done

    log_success "Deleted $DELETE_COUNT old pg_dump backup(s)"
else
    log "Current pg_dump backup count: $BACKUP_COUNT (within retention limit)"
fi

# ===================================================================
# Part 2: Docker Volume Backup
# ===================================================================

log ""
log "==================================================================="
log "Starting Docker Volume Backup..."
log "==================================================================="

# Find the postgres volume
POSTGRES_VOLUME=$(docker volume ls --filter "name=postgres" --format "{{.Name}}" | grep -i "langfuse" | head -1)

if [ -z "$POSTGRES_VOLUME" ]; then
    log_warning "Could not find Langfuse postgres volume. Skipping volume backup."
    log "Available volumes with 'postgres' in name:"
    docker volume ls --filter "name=postgres" --format "table {{.Name}}\t{{.Driver}}"
else
    log "Found postgres volume: $POSTGRES_VOLUME"

    # Backup the volume using a temporary container
    VOLUME_BACKUP_PATH="$VOLUME_BACKUP_DIR/$VOLUME_BACKUP_FILE"

    log "Creating volume backup (this may take a few minutes)..."
    docker run --rm \
        -v "$POSTGRES_VOLUME:/data:ro" \
        -v "$VOLUME_BACKUP_DIR:/backup" \
        alpine \
        tar czf "/backup/$VOLUME_BACKUP_FILE" -C /data .

    # Check if volume backup was successful
    if [ $? -ne 0 ]; then
        log_error "Volume backup failed"
    elif [ ! -f "$VOLUME_BACKUP_PATH" ]; then
        log_error "Volume backup file was not created"
    else
        VOLUME_BACKUP_SIZE=$(stat -f%z "$VOLUME_BACKUP_PATH" 2>/dev/null || stat -c%s "$VOLUME_BACKUP_PATH" 2>/dev/null)
        if [ "$VOLUME_BACKUP_SIZE" -eq 0 ]; then
            log_error "Volume backup file is empty"
            rm "$VOLUME_BACKUP_PATH"
        else
            VOLUME_BACKUP_SIZE_MB=$(echo "scale=2; $VOLUME_BACKUP_SIZE / 1048576" | bc)
            log_success "Volume backup created: $VOLUME_BACKUP_FILE (${VOLUME_BACKUP_SIZE_MB} MB)"

            # Manage volume backup retention
            log "Managing volume backup retention (keeping last $RETENTION_COUNT backups)..."
            VOLUME_BACKUP_COUNT=$(ls -1 "$VOLUME_BACKUP_DIR"/langfuse_volume_*.tar.gz 2>/dev/null | wc -l)

            if [ "$VOLUME_BACKUP_COUNT" -gt "$RETENTION_COUNT" ]; then
                DELETE_COUNT=$((VOLUME_BACKUP_COUNT - RETENTION_COUNT))
                log "Found $VOLUME_BACKUP_COUNT volume backups, deleting oldest $DELETE_COUNT..."

                ls -1t "$VOLUME_BACKUP_DIR"/langfuse_volume_*.tar.gz | tail -n "$DELETE_COUNT" | while read -r old_backup; do
                    log "Deleting old volume backup: $(basename "$old_backup")"
                    rm "$old_backup"
                done

                log_success "Deleted $DELETE_COUNT old volume backup(s)"
            else
                log "Current volume backup count: $VOLUME_BACKUP_COUNT (within retention limit)"
            fi
        fi
    fi
fi

# Summary
log ""
log "==================================================================="
log_success "All backups completed successfully"
log "==================================================================="
log "pg_dump backup:"
log "  Location: $BACKUP_PATH"
log "  Size: ${BACKUP_SIZE_MB} MB"
log "  Total count: $(ls -1 "$BACKUP_DIR"/langfuse_backup_*.sql.gz 2>/dev/null | wc -l)"
log ""
if [ -n "$POSTGRES_VOLUME" ] && [ -f "$VOLUME_BACKUP_PATH" ]; then
    log "Volume backup:"
    log "  Location: $VOLUME_BACKUP_PATH"
    log "  Size: ${VOLUME_BACKUP_SIZE_MB} MB"
    log "  Total count: $(ls -1 "$VOLUME_BACKUP_DIR"/langfuse_volume_*.tar.gz 2>/dev/null | wc -l)"
fi
log "==================================================================="

exit 0
