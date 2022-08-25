#!/bin/bash
docker exec -e COLUMNS="`tput cols`" -e LINES="`tput lines`" -ti pt5 /bin/bash
