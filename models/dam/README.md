## mmgroundingdino

Lightweight vision server for [DAM](https://github.com/NVlabs/describe-anything)

### Usage

```bash
docker compose build
```

```bash
export DAM_TOKEN="super_secret_access_token"
export DAM_PORT=9000
chmod +x docker/include/entrypoint.sh
```

[Optional] To preload the server with a flavor of the model:

```bash
export DAM_PRELOAD_FLAVOR="3B"
```

```bash
docker compose up
```