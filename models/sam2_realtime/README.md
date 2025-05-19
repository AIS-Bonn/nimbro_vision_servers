## mmgroundingdino

Lightweight vision server for the [SAM2-Realtime Model](https://github.com/Gy920/segment-anything-2-real-time)

### Usage

```bash
docker compose build
```

```bash
export SAM2_REALTIME_TOKEN="super_secret_access_token"
export SAM2_REALTIME_PORT=9000
chmod +x docker/include/entrypoint.sh
```

[Optional] To preload the server with a flavor of the model:

```bash
export SAM2_REALTIME_PRELOAD_FLAVOR="large"
```

```bash
docker compose up
```