# Cocos Pack Inspector

This repository provides a simple viewer for Cocos Creator 2.x bundles. The `test.py` script can decode the packed JSON and display node hierarchies, component data and asset references.

A small example bundle is bundled under `main_v2` for experimentation.

## Quick start

Run the inspector using the example configuration:

```bash
python test.py main_v2/config.c2200.json
```

### Optional flags

- `--assets`  Show the full asset table instead of only referenced assets.
- `--test`    Run the internal component decoder test suite.
