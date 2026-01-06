# Langfuse Docker Backup & Restore Guide

This guide shows you how to back up and restore your Langfuse Docker instance running locally.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Manual Backup](#manual-backup)
- [Automated Backups (Cron)](#automated-backups-cron)
- [Restore from Backup](#restore-from-backup)
- [Troubleshooting](#troubleshooting)
- [Advanced Topics](#advanced-topics)

## Overview

This backup solution uses a **dual-backup approach** for maximum safety:

### Two Backup Methods

1. **pg_dump (Logical Backup)**
   - Exports database as SQL statements
   - Portable across PostgreSQL versions
   - Smaller file size
   - Best for: General restores, migrations, disaster recovery

2. **Docker Volume Backup (Physical Backup)**
   - Complete snapshot of database volume
   - Includes all PostgreSQL config files
   - Faster restore
   - Best for: Quick recovery, exact state restoration

### Features

- ✓ Backs up PostgreSQL database using BOTH methods
- ✓ Creates compressed backups (`.sql.gz` and `.tar.gz` formats)
- ✓ Stores backups locally in `./langfuse-backups/` (relative to project directory)
- ✓ Automatically keeps last 7 of each backup type (configurable)
- ✓ Logs all operations
- ✓ Can run manually or automatically via cron
- ✓ No downtime required

### What Gets Backed Up

| Component | pg_dump | Volume Backup | Contents |
|-----------|---------|---------------|----------|
| PostgreSQL Data | ✓ Yes | ✓ Yes | Users, projects, prompts, settings, trace metadata |
| PostgreSQL Config | ✗ No | ✓ Yes | postgresql.conf, pg_hba.conf, etc. |
| ClickHouse | ✗ No* | ✗ No* | Analytics and trace data |
| MinIO | ✗ No* | ✗ No* | Media files, event uploads, exports |
| Redis | ✗ No | ✗ No | Cache and queues (ephemeral) |

*See [Advanced Topics](#advanced-topics) for backing up ClickHouse and MinIO

## Prerequisites

- Docker and docker-compose installed
- Langfuse running via docker-compose
- Bash shell (macOS, Linux, WSL)
- At least 1GB free disk space

## Quick Start

### 1. Run Your First Backup

```bash
cd /Users/drucev/projects/OpenAIAgentsSDK
./backup_langfuse.sh
```

You should see output like:
```
[2026-01-04 10:30:00] Creating backup directory: /Users/drucev/langfuse-backups
[2026-01-04 10:30:00] Looking for Langfuse postgres container...
[2026-01-04 10:30:00] Found postgres container: langfuse-postgres-1
[2026-01-04 10:30:01] Starting backup of database 'postgres'...
[2026-01-04 10:30:15] SUCCESS: Backup created: langfuse_backup_20260104_103000.sql.gz (42.5 MB)
[2026-01-04 10:30:15] Managing pg_dump backup retention (keeping last 7 backups)...

===================================================================
Starting Docker Volume Backup...
===================================================================
[2026-01-04 10:30:16] Found postgres volume: langfuse_postgres_data
[2026-01-04 10:30:16] Creating volume backup (this may take a few minutes)...
[2026-01-04 10:30:45] SUCCESS: Volume backup created: langfuse_volume_20260104_103000.tar.gz (128.3 MB)

===================================================================
SUCCESS: All backups completed successfully
===================================================================
pg_dump backup:
  Location: /Users/drucev/langfuse-backups/langfuse_backup_20260104_103000.sql.gz
  Size: 42.5 MB
  Total count: 1

Volume backup:
  Location: /Users/drucev/langfuse-backups/volumes/langfuse_volume_20260104_103000.tar.gz
  Size: 128.3 MB
  Total count: 1
===================================================================
```

### 2. Verify Your Backups

```bash
# View pg_dump backups
ls -lh /Users/drucev/projects/OpenAIAgentsSDK/langfuse-backups/*.sql.gz

# View volume backups
ls -lh /Users/drucev/projects/OpenAIAgentsSDK/langfuse-backups/volumes/*.tar.gz

# View all backups (if you have tree installed)
tree /Users/drucev/projects/OpenAIAgentsSDK/langfuse-backups/

# Or use ls -R
ls -lhR /Users/drucev/projects/OpenAIAgentsSDK/langfuse-backups/
```

You should see:
```
/Users/drucev/projects/OpenAIAgentsSDK/langfuse-backups/
├── langfuse_backup_20260104_103000.sql.gz  (42.5 MB)
├── backup.log
└── volumes/
    └── langfuse_volume_20260104_103000.tar.gz  (128.3 MB)
```

### 3. Set Up Automated Daily Backups

See [Automated Backups](#automated-backups-cron) section below.

## Manual Backup

### Running a Backup

```bash
cd /Users/drucev/projects/OpenAIAgentsSDK
./backup_langfuse.sh
```

### What Happens During Backup

The script performs TWO backups sequentially:

**Part 1: pg_dump Backup**
1. Verifies Docker is running and finds postgres container
2. Creates backup directories if they don't exist
3. Checks available disk space
4. Exports database using `pg_dump` inside the postgres container
5. Compresses the SQL dump with gzip
6. Verifies backup file exists and has content
7. Rotates old pg_dump backups (keeps last 7)

**Part 2: Volume Backup**
8. Finds the postgres Docker volume
9. Creates temporary Alpine container to access the volume
10. Creates compressed tar archive of entire volume
11. Verifies volume backup file exists and has content
12. Rotates old volume backups (keeps last 7)
13. Reports summary with both backup sizes and locations

### Backup File Naming

Both backup types use the same timestamp:

**pg_dump backups**:
```
langfuse_backup_YYYYMMDD_HHMMSS.sql.gz
```

**Volume backups**:
```
langfuse_volume_YYYYMMDD_HHMMSS.tar.gz
```

Example: `langfuse_backup_20260104_143022.sql.gz`
- Date: January 4, 2026
- Time: 2:30:22 PM

Both backups from the same run share the same timestamp, making it easy to match them.

## Automated Backups (Cron)

### Set Up Daily Backups at 2:00 AM

1. **Open your crontab**:
   ```bash
   crontab -e
   ```

2. **Add this line** (press `i` to enter insert mode in vim):
   ```
   0 2 * * * /Users/drucev/projects/OpenAIAgentsSDK/backup_langfuse.sh >> /Users/drucev/projects/OpenAIAgentsSDK/langfuse-backups/backup.log 2>&1
   ```

3. **Save and exit** (press `Esc`, type `:wq`, press Enter)

4. **Verify the cron job**:
   ```bash
   crontab -l
   ```

   You should see your backup line listed.

### Cron Schedule Explanation

```
0 2 * * *
│ │ │ │ │
│ │ │ │ └─── Day of week (0-7, where 0 and 7 = Sunday)
│ │ │ └───── Month (1-12)
│ │ └─────── Day of month (1-31)
│ └───────── Hour (0-23)
└─────────── Minute (0-59)
```

**Common schedules**:
- `0 2 * * *` - Daily at 2:00 AM
- `0 2 * * 0` - Weekly on Sunday at 2:00 AM
- `0 */6 * * *` - Every 6 hours
- `0 2 1 * *` - Monthly on the 1st at 2:00 AM

### View Backup Logs

```bash
tail -f /Users/drucev/projects/OpenAIAgentsSDK/langfuse-backups/backup.log
```

## Restore from Backup

You have **two restore options**:

1. **Restore from pg_dump** - Recommended for most situations
2. **Restore from volume backup** - Faster, but requires exact PostgreSQL version match

Choose the method that best fits your situation.

---

### Option A: Restore from pg_dump (Recommended)

This is the most flexible restore method. Use this when:
- Migrating to a different server
- Upgrading PostgreSQL version
- Restoring specific tables or data
- You're not sure which method to use

#### Step-by-Step Process

##### 1. Stop Langfuse Services

```bash
cd /Users/drucev/projects/OpenAIAgentsSDK/langfuse
docker-compose down
```

This stops all services (web, worker, postgres, clickhouse, redis, minio).

##### 2. Start Only PostgreSQL

```bash
docker-compose up -d postgres
```

Wait a few seconds for postgres to be ready:
```bash
docker-compose logs postgres
```

Look for: `database system is ready to accept connections`

##### 3. Choose Your Backup File

List available pg_dump backups:
```bash
ls -lht /Users/drucev/projects/OpenAIAgentsSDK/langfuse-backups/langfuse_backup_*.sql.gz | head -10
```

This shows the 10 most recent backups, newest first.

##### 4. Restore the Database

Replace `YYYYMMDD_HHMMSS` with your chosen backup timestamp:

```bash
gunzip -c /Users/drucev/projects/OpenAIAgentsSDK/langfuse-backups/langfuse_backup_YYYYMMDD_HHMMSS.sql.gz | \
  docker exec -i langfuse-postgres-1 psql -U postgres postgres
```

**Example**:
```bash
gunzip -c /Users/drucev/projects/OpenAIAgentsSDK/langfuse-backups/langfuse_backup_20260104_020000.sql.gz | \
  docker exec -i langfuse-postgres-1 psql -U postgres postgres
```

You'll see SQL statements scrolling by. This is normal.

##### 5. Start All Services

```bash
docker-compose up -d
```

##### 6. Verify Restoration

Open Langfuse in your browser:
```
http://localhost:3000
```

Log in and verify your data is restored.

#### Quick pg_dump Restore (One-Liner)

```bash
cd /Users/drucev/projects/OpenAIAgentsSDK/langfuse && \
docker-compose down && \
docker-compose up -d postgres && \
sleep 5 && \
gunzip -c /Users/drucev/projects/OpenAIAgentsSDK/langfuse-backups/langfuse_backup_YYYYMMDD_HHMMSS.sql.gz | \
  docker exec -i langfuse-postgres-1 psql -U postgres postgres && \
docker-compose up -d
```

#### Advanced SQL Dump Restore Options

##### Option 1: Clean Restore (Recommended)

This method drops and recreates the database for a completely clean restore:

```bash
# 1. Stop all services
cd /Users/drucev/projects/OpenAIAgentsSDK/langfuse
docker-compose down

# 2. Start only postgres
docker-compose up -d postgres
sleep 5

# 3. Drop existing database (WARNING: This deletes all current data!)
docker exec -i langfuse-postgres-1 psql -U postgres -c "DROP DATABASE IF EXISTS postgres;"

# 4. Recreate database
docker exec -i langfuse-postgres-1 psql -U postgres -c "CREATE DATABASE postgres;"

# 5. Restore from backup
gunzip -c /Users/drucev/projects/OpenAIAgentsSDK/langfuse-backups/langfuse_backup_YYYYMMDD_HHMMSS.sql.gz | \
  docker exec -i langfuse-postgres-1 psql -U postgres postgres

# 6. Start all services
docker-compose up -d
```

##### Option 2: Restore to a Different Database (Testing)

Restore to a test database to verify backup integrity without affecting production:

```bash
# 1. Create test database
docker exec -i langfuse-postgres-1 psql -U postgres -c "CREATE DATABASE langfuse_test;"

# 2. Restore backup to test database
gunzip -c /Users/drucev/projects/OpenAIAgentsSDK/langfuse-backups/langfuse_backup_YYYYMMDD_HHMMSS.sql.gz | \
  docker exec -i langfuse-postgres-1 psql -U postgres langfuse_test

# 3. Verify tables exist
docker exec -i langfuse-postgres-1 psql -U postgres langfuse_test -c "\dt"

# 4. Check row counts
docker exec -i langfuse-postgres-1 psql -U postgres langfuse_test -c \
  "SELECT schemaname,tablename,n_live_tup FROM pg_stat_user_tables ORDER BY n_live_tup DESC;"

# 5. Drop test database when done
docker exec -i langfuse-postgres-1 psql -U postgres -c "DROP DATABASE langfuse_test;"
```

##### Option 3: Restore Specific Tables Only

If you only need to restore specific tables (advanced users):

```bash
# 1. Extract specific table from backup
gunzip -c /Users/drucev/projects/OpenAIAgentsSDK/langfuse-backups/langfuse_backup_YYYYMMDD_HHMMSS.sql.gz | \
  grep -A 10000 "Table structure for table \`prompts\`" | \
  grep -B 10000 "Table structure for table \`" | head -n -1 > /tmp/prompts_table.sql

# 2. Restore just that table
docker exec -i langfuse-postgres-1 psql -U postgres postgres < /tmp/prompts_table.sql

# 3. Clean up
rm /tmp/prompts_table.sql
```

**Note**: This is complex and may require manual editing of the SQL file. Only use if you know what you're doing.

#### Verifying Your Restore

After restoring, verify the data is correct:

```bash
# 1. Check database size
docker exec -i langfuse-postgres-1 psql -U postgres postgres -c \
  "SELECT pg_size_pretty(pg_database_size('postgres'));"

# 2. List all tables
docker exec -i langfuse-postgres-1 psql -U postgres postgres -c "\dt"

# 3. Check table row counts
docker exec -i langfuse-postgres-1 psql -U postgres postgres -c \
  "SELECT schemaname,tablename,n_live_tup FROM pg_stat_user_tables ORDER BY n_live_tup DESC;"

# 4. Check most recent data (example for prompts)
docker exec -i langfuse-postgres-1 psql -U postgres postgres -c \
  "SELECT name, version, created_at FROM prompts ORDER BY created_at DESC LIMIT 5;"

# 5. Verify in Langfuse UI
# Open http://localhost:3000 and check:
# - Can log in
# - Projects are visible
# - Prompts are visible
# - Trace data appears correct
```

#### Troubleshooting SQL Dump Restores

##### "ERROR: database does not exist"

**Problem**: The database 'postgres' doesn't exist in the container.

**Solution**:
```bash
# Create the database first
docker exec -i langfuse-postgres-1 psql -U postgres -c "CREATE DATABASE postgres;"

# Then restore
gunzip -c /Users/drucev/projects/OpenAIAgentsSDK/langfuse-backups/langfuse_backup_YYYYMMDD_HHMMSS.sql.gz | \
  docker exec -i langfuse-postgres-1 psql -U postgres postgres
```

##### "ERROR: must be owner of table" or Permission Errors

**Problem**: Restore is trying to change ownership to a user that doesn't exist.

**Solution**: Use `--no-owner` flag with pg_restore, or edit the SQL dump:

```bash
# Filter out ownership commands
gunzip -c /Users/drucev/projects/OpenAIAgentsSDK/langfuse-backups/langfuse_backup_YYYYMMDD_HHMMSS.sql.gz | \
  grep -v "ALTER TABLE.*OWNER TO" | \
  docker exec -i langfuse-postgres-1 psql -U postgres postgres
```

##### "ERROR: relation already exists"

**Problem**: Tables already exist from previous data.

**Solutions**:

1. **Drop database first** (clean restore):
   ```bash
   docker exec -i langfuse-postgres-1 psql -U postgres -c "DROP DATABASE postgres;"
   docker exec -i langfuse-postgres-1 psql -U postgres -c "CREATE DATABASE postgres;"
   # Then restore
   ```

2. **Drop only conflicting tables**:
   ```bash
   docker exec -i langfuse-postgres-1 psql -U postgres postgres -c "DROP TABLE IF EXISTS tablename CASCADE;"
   ```

3. **Use --clean flag** in future backups:
   ```bash
   # When creating backups, add --clean flag
   docker exec langfuse-postgres-1 pg_dump -U postgres --clean postgres | gzip > backup.sql.gz
   ```

##### Restore is Very Slow

**Problem**: Large database taking a long time to restore.

**Solutions**:

1. **Disable triggers during restore** (faster but riskier):
   ```bash
   echo "SET session_replication_role = replica;" > /tmp/restore.sql
   gunzip -c /Users/drucev/projects/OpenAIAgentsSDK/langfuse-backups/langfuse_backup_YYYYMMDD_HHMMSS.sql.gz >> /tmp/restore.sql
   echo "SET session_replication_role = DEFAULT;" >> /tmp/restore.sql

   docker exec -i langfuse-postgres-1 psql -U postgres postgres < /tmp/restore.sql
   rm /tmp/restore.sql
   ```

2. **Increase PostgreSQL work_mem**:
   ```bash
   docker exec -i langfuse-postgres-1 psql -U postgres -c \
     "ALTER SYSTEM SET work_mem = '256MB';"
   docker restart langfuse-postgres-1
   ```

3. **Monitor progress**:
   ```bash
   # In one terminal, start restore
   gunzip -c backup.sql.gz | docker exec -i langfuse-postgres-1 psql -U postgres postgres

   # In another terminal, watch progress
   docker exec -i langfuse-postgres-1 psql -U postgres postgres -c \
     "SELECT count(*) FROM pg_stat_activity WHERE state = 'active';"
   ```

##### Backup File is Corrupted

**Problem**: `gunzip: invalid compressed data--format violated`

**Solutions**:

1. **Test file integrity**:
   ```bash
   gunzip -t /Users/drucev/projects/OpenAIAgentsSDK/langfuse-backups/langfuse_backup_YYYYMMDD_HHMMSS.sql.gz
   ```

2. **Try alternative decompression**:
   ```bash
   gzip -dc /Users/drucev/projects/OpenAIAgentsSDK/langfuse-backups/langfuse_backup_YYYYMMDD_HHMMSS.sql.gz | \
     docker exec -i langfuse-postgres-1 psql -U postgres postgres
   ```

3. **Use an older backup** that isn't corrupted.

##### Out of Disk Space During Restore

**Problem**: Not enough space to decompress and restore.

**Solution**: Stream directly without saving decompressed file (already default in our commands).

If still out of space:
```bash
# Check available space
df -h /var/lib/docker

# Clean up Docker to free space
docker system prune -a --volumes

# Or expand disk if on cloud/VM
```

#### Best Practices for SQL Dump Restores

1. **Always test your backups** - Periodically restore to a test database to ensure backups work
2. **Restore to staging first** - Never restore directly to production without testing
3. **Take a backup before restoring** - Create a safety backup of current state
4. **Verify data integrity** - Check row counts and recent data after restore
5. **Monitor disk space** - Ensure enough space for both backup and restored data
6. **Document your process** - Keep notes on any special steps needed for your setup
7. **Use clean restores** - Drop/recreate database for cleanest results

---

### Option B: Restore from Volume Backup (Faster)

This method is faster but less flexible. Use this when:
- You need the fastest possible restore
- You're restoring to the same PostgreSQL version
- You want to preserve all PostgreSQL configuration

**Warning**: This will completely replace the postgres volume. Make sure you have the right backup!

#### Step-by-Step Process

##### 1. Stop All Langfuse Services

```bash
cd /Users/drucev/projects/OpenAIAgentsSDK/langfuse
docker-compose down
```

##### 2. Choose Your Volume Backup

List available volume backups:
```bash
ls -lht /Users/drucev/projects/OpenAIAgentsSDK/langfuse-backups/volumes/langfuse_volume_*.tar.gz | head -10
```

##### 3. Remove Existing Volume

**Warning**: This deletes all current data!

```bash
docker volume rm langfuse_postgres_data
```

If you get an error that the volume is in use, make sure all containers are stopped:
```bash
docker ps -a | grep postgres
docker rm -f <container_name>  # if any are listed
```

##### 4. Create New Empty Volume

```bash
docker volume create langfuse_postgres_data
```

##### 5. Restore Volume from Backup

Replace `YYYYMMDD_HHMMSS` with your chosen backup timestamp:

```bash
docker run --rm \
  -v langfuse_postgres_data:/data \
  -v /Users/drucev/projects/OpenAIAgentsSDK/langfuse-backups/volumes:/backup \
  alpine \
  sh -c "cd /data && tar xzf /backup/langfuse_volume_YYYYMMDD_HHMMSS.tar.gz"
```

**Example**:
```bash
docker run --rm \
  -v langfuse_postgres_data:/data \
  -v /Users/drucev/projects/OpenAIAgentsSDK/langfuse-backups/volumes:/backup \
  alpine \
  sh -c "cd /data && tar xzf /backup/langfuse_volume_20260104_020000.tar.gz"
```

##### 6. Start All Services

```bash
docker-compose up -d
```

##### 7. Verify Restoration

Open Langfuse in your browser:
```
http://localhost:3000
```

Log in and verify your data is restored.

#### Quick Volume Restore (One-Liner)

**Warning**: This destroys existing data!

```bash
cd /Users/drucev/projects/OpenAIAgentsSDK/langfuse && \
docker-compose down && \
docker volume rm langfuse_postgres_data && \
docker volume create langfuse_postgres_data && \
docker run --rm -v langfuse_postgres_data:/data -v /Users/drucev/projects/OpenAIAgentsSDK/langfuse-backups/volumes:/backup alpine sh -c "cd /data && tar xzf /backup/langfuse_volume_YYYYMMDD_HHMMSS.tar.gz" && \
docker-compose up -d
```

---

### Which Restore Method Should I Use?

| Scenario | Recommended Method |
|----------|-------------------|
| General restore | pg_dump |
| Migrating to new server | pg_dump |
| Upgrading PostgreSQL | pg_dump |
| Need fastest restore | Volume backup |
| Exact state restoration | Volume backup |
| Disaster recovery | Either (try pg_dump first) |
| Not sure | pg_dump |

## Troubleshooting

### "Could not find Langfuse postgres container"

**Problem**: Script can't find the postgres container.

**Solutions**:
1. Check if containers are running:
   ```bash
   docker ps
   ```

2. Look for a container with "postgres" in the name. If you see it, note the exact name.

3. Start Langfuse if it's not running:
   ```bash
   cd /Users/drucev/projects/OpenAIAgentsSDK/langfuse
   docker-compose up -d
   ```

4. If container has a different name, edit the script and update the container search pattern.

### "Docker is not running"

**Problem**: Docker daemon is not started.

**Solutions**:
- macOS: Open Docker Desktop application
- Linux: `sudo systemctl start docker`

### Backup File is Empty (0 bytes)

**Problem**: Backup completed but file is empty.

**Possible causes**:
1. Database connection failed
2. Insufficient permissions
3. Database is empty

**Solutions**:
1. Check postgres container logs:
   ```bash
   docker logs langfuse-postgres-1
   ```

2. Test manual connection:
   ```bash
   docker exec -it langfuse-postgres-1 psql -U postgres postgres -c "\dt"
   ```

3. Verify database has data:
   ```bash
   docker exec langfuse-postgres-1 psql -U postgres postgres -c "SELECT COUNT(*) FROM pg_tables WHERE schemaname='public';"
   ```

### "Permission denied" When Running Script

**Problem**: Script is not executable.

**Solution**:
```bash
chmod +x /Users/drucev/projects/OpenAIAgentsSDK/backup_langfuse.sh
```

### Cron Job Not Running

**Problem**: Automated backups aren't happening.

**Debugging steps**:

1. **Check cron is running**:
   ```bash
   # macOS
   sudo launchctl list | grep cron

   # Linux
   sudo systemctl status cron
   ```

2. **Verify crontab entry**:
   ```bash
   crontab -l
   ```

3. **Check cron logs** (macOS):
   ```bash
   log show --predicate 'process == "cron"' --last 1d
   ```

4. **Test the script manually**:
   ```bash
   /Users/drucev/projects/OpenAIAgentsSDK/backup_langfuse.sh
   ```

5. **Check backup log**:
   ```bash
   tail -n 50 /Users/drucev/projects/OpenAIAgentsSDK/langfuse-backups/backup.log
   ```

6. **Give cron full disk access** (macOS Catalina+):
   - System Preferences → Security & Privacy → Privacy → Full Disk Access
   - Add `/usr/sbin/cron`

### Low Disk Space Warning

**Problem**: Script warns about low disk space.

**Solutions**:

1. **Check current usage**:
   ```bash
   du -sh /Users/drucev/projects/OpenAIAgentsSDK/langfuse-backups/
   ```

2. **Reduce retention count** (edit script, change `RETENTION_COUNT=7` to a lower number)

3. **Move backups to external drive**:
   ```bash
   mv /Users/drucev/projects/OpenAIAgentsSDK/langfuse-backups /Volumes/ExternalDrive/langfuse-backups
   ln -s /Volumes/ExternalDrive/langfuse-backups /Users/drucev/projects/OpenAIAgentsSDK/langfuse-backups
   ```

4. **Compress old backups further** (already gzipped, but you can use better compression):
   ```bash
   cd /Users/drucev/projects/OpenAIAgentsSDK/langfuse-backups
   for f in langfuse_backup_*.sql.gz; do
     gunzip "$f"
     xz -9 "${f%.gz}"
   done
   ```

## Advanced Topics

### Changing Backup Location

Edit the script and change the `BACKUP_DIR` variable:

```bash
# In backup_langfuse.sh, line 14:
BACKUP_DIR="/path/to/your/backup/location"
```

### Changing Retention Policy

Edit the script and change the `RETENTION_COUNT` variable:

```bash
# In backup_langfuse.sh, line 16:
RETENTION_COUNT=14  # Keep last 14 backups instead of 7
```

### Backing Up ClickHouse (Analytics Database)

ClickHouse stores analytics and trace data. If you need to back it up:

```bash
# Create ClickHouse backup directory
mkdir -p /Users/drucev/projects/OpenAIAgentsSDK/langfuse-backups/clickhouse

# Backup ClickHouse database
docker exec langfuse-clickhouse-1 clickhouse-client --query="BACKUP DATABASE default TO Disk('default', 'backup.zip')"

# Copy backup out of container
docker cp langfuse-clickhouse-1:/var/lib/clickhouse/backup.zip /Users/drucev/projects/OpenAIAgentsSDK/langfuse-backups/clickhouse/clickhouse_backup_$(date +%Y%m%d_%H%M%S).zip
```

**Note**: ClickHouse backups can be very large if you have a lot of trace data.

### Backing Up MinIO (File Storage)

MinIO stores uploaded files and media. To back it up:

```bash
# Install MinIO client (mc)
brew install minio/stable/mc  # macOS
# OR
wget https://dl.min.io/client/mc/release/linux-amd64/mc  # Linux

# Configure MinIO client
mc alias set langfuse-local http://localhost:9090 minio miniosecret

# Create backup directory
mkdir -p /Users/drucev/projects/OpenAIAgentsSDK/langfuse-backups/minio

# Mirror MinIO bucket
mc mirror langfuse-local/langfuse /Users/drucev/projects/OpenAIAgentsSDK/langfuse-backups/minio/
```

### Encrypting Backups

For sensitive data, encrypt backups with GPG:

```bash
# Encrypt a backup
gpg --symmetric --cipher-algo AES256 /Users/drucev/projects/OpenAIAgentsSDK/langfuse-backups/langfuse_backup_YYYYMMDD_HHMMSS.sql.gz

# This creates: langfuse_backup_YYYYMMDD_HHMMSS.sql.gz.gpg

# Decrypt when restoring
gpg --decrypt /Users/drucev/projects/OpenAIAgentsSDK/langfuse-backups/langfuse_backup_YYYYMMDD_HHMMSS.sql.gz.gpg | \
  docker exec -i langfuse-postgres-1 psql -U postgres postgres
```

### Remote Backup to Cloud Storage

#### Upload to AWS S3

```bash
# Install AWS CLI
brew install awscli  # macOS

# Configure AWS credentials
aws configure

# Add to backup script or run after backup:
aws s3 sync /Users/drucev/projects/OpenAIAgentsSDK/langfuse-backups/ s3://your-bucket-name/langfuse-backups/
```

#### Upload to Dropbox

```bash
# Install rclone
brew install rclone  # macOS

# Configure Dropbox
rclone config

# Sync backups
rclone sync /Users/drucev/projects/OpenAIAgentsSDK/langfuse-backups/ dropbox:langfuse-backups/
```

### Testing Backup Integrity

Periodically test that your backups can be restored:

```bash
# 1. Create a test restore environment
docker run -d --name postgres-test -e POSTGRES_PASSWORD=test postgres:17

# 2. Wait for postgres to start
sleep 5

# 3. Restore backup to test container
gunzip -c /Users/drucev/projects/OpenAIAgentsSDK/langfuse-backups/langfuse_backup_YYYYMMDD_HHMMSS.sql.gz | \
  docker exec -i postgres-test psql -U postgres postgres

# 4. Verify tables exist
docker exec postgres-test psql -U postgres postgres -c "\dt"

# 5. Clean up
docker stop postgres-test
docker rm postgres-test
```

### Email Notifications on Failure

Add email notifications when backups fail. Edit the script and add after line 1:

```bash
# Add at top of script
NOTIFY_EMAIL="your-email@example.com"

# Add this function
send_notification() {
    local subject="$1"
    local message="$2"
    echo "$message" | mail -s "$subject" "$NOTIFY_EMAIL"
}

# Replace exit 1 calls with:
send_notification "Langfuse Backup Failed" "Backup failed at $(date). Check logs."
exit 1
```

### Differential/Incremental Backups

For very large databases, consider incremental backups using PostgreSQL WAL archiving:

```bash
# Enable WAL archiving in postgresql.conf
docker exec langfuse-postgres-1 bash -c "echo 'wal_level = replica' >> /var/lib/postgresql/data/postgresql.conf"
docker exec langfuse-postgres-1 bash -c "echo 'archive_mode = on' >> /var/lib/postgresql/data/postgresql.conf"
docker exec langfuse-postgres-1 bash -c "echo 'archive_command = '\''cp %p /archive/%f'\''' >> /var/lib/postgresql/data/postgresql.conf"

# Restart postgres
docker restart langfuse-postgres-1
```

This is advanced and requires additional setup. See PostgreSQL documentation for details.

## Related Files

- `backup_langfuse.sh` - Backup script
- `upload_prompts_to_langfuse.py` - Upload prompts to Langfuse
- `UPLOAD_PROMPTS_GUIDE.md` - Guide for uploading prompts
- `langfuse/docker-compose.yml` - Docker Compose configuration

## Support

If you encounter issues:

1. Check the [Troubleshooting](#troubleshooting) section
2. Review backup logs: `/Users/drucev/projects/OpenAIAgentsSDK/langfuse-backups/backup.log`
3. Check Langfuse documentation: https://langfuse.com/docs
4. Open an issue on GitHub: https://github.com/langfuse/langfuse

## Best Practices

1. **Test restores regularly** - Backups are only useful if they can be restored
2. **Store backups off-site** - Protect against hardware failure
3. **Monitor disk space** - Ensure you have enough space for backups
4. **Encrypt sensitive data** - Use GPG or similar for encryption
5. **Document your process** - Keep notes on restore procedures
6. **Verify backup integrity** - Check that backup files aren't corrupted
7. **Keep multiple generations** - Don't rely on just the latest backup

## Backup Checklist

- [ ] Backup script created and tested
- [ ] Cron job configured and verified
- [ ] Backup logs reviewed
- [ ] Test restore performed successfully
- [ ] Backup retention policy configured
- [ ] Off-site backup configured (optional)
- [ ] Email notifications set up (optional)
- [ ] Team knows restore procedure
- [ ] Backup schedule documented
- [ ] Disaster recovery plan documented
