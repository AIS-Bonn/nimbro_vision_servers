## mmgroundingdino

Lightweight vision server for [Florence-2](https://arxiv.org/abs/2311.06242)

### Usage

```bash
docker compose build
```

```bash
export FLORENCE2_TOKEN="super_secret_access_token"
export FLORENCE2_PORT=9000
chmod +x docker/include/entrypoint.sh
```

[Optional] To preload the server with a flavor of the model:

```bash
export FLORENCE2_PRELOAD_FLAVOR="large"
```

```bash
docker compose up
```