#!/bin/bash
echo "Docker starting pt5."

# Make sure we aren't running...
echo "Bringing Docker down..."
docker-compose down

# Up time
echo "PRODUCTION ENVIRONMENT ENABLED!"
export UID=${UID}
export GID=${GID}
docker-compose -f dev-docker-compose.yml up -d
