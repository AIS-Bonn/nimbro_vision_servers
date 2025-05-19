## mmgroundingdino

Lightweight vision server for [KOSMOS-2](https://arxiv.org/abs/2306.14824)

### Usage

```bash
docker compose build
```

```bash
export KOSMOS2_TOKEN="super_secret_access_token"
export KOSMOS2_PORT=9000
chmod +x docker/include/entrypoint.sh
```

[Optional] To preload the server with a flavor of the model:

```bash
export KOSMOS2_PRELOAD_FLAVOR="patch14-224"
```

```bash
docker compose up
```