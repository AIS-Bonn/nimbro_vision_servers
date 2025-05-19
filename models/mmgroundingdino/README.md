## mmgroundingdino

Lightweight vision server for the [MM Grounding DINO](https://github.com/open-mmlab/mmdetection/blob/main/configs/mm_grounding_dino/README.md) and [LLMDet](https://github.com/iSEE-Laboratory/LLMDet) models.

### Usage

```bash
docker compose build
```

```bash
export MMGROUNDINGDINO_TOKEN="super_secret_access_token"
export MMGROUNDINGDINO_PORT=9000
chmod +x docker/include/entrypoint.sh
```

[Optional] To preload the server with a flavor of the model:

```bash
export MMGROUNDINGDINO_PRELOAD_FLAVOR="large_zeroshot"
```

```bash
docker compose up
```