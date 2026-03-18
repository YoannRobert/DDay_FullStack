---
title: MLflow DDay
emoji: ⚡️
colorFrom: yellow
colorTo: red
sdk: docker
pinned: false
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

```

### Fonctionnement
```
push from main (files in spaces/mlflow/)
        ↓
GitHub Action is triggered
        ↓
git init + commit of the contents from spaces/mlflow/
        ↓
force-push to huggingface.co/spaces/...
        ↓
HF Space automatically rebuilds the Docker
