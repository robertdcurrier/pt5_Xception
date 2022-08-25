#!/bin/bash
echo "Docker starting ImageAI..."

# Make sure we aren't running...
echo "Bringing Docker down..."
docker-compose down

# Up time
echo "PRODUCTION ENVIRONMENT ENABLED!"
export UID=${UID}
export GID=${GID}
docker-compose -f docker-compose.yml up -d
