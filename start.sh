#!/usr/bin/env bash
gunicorn app:app.server --bind 0.0.0.0:$PORT
