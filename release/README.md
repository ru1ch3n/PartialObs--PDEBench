# Publishing a dataset tier

Generated arrays do not belong in Git history. Publish them in a versioned
dataset archive or GitHub release, then commit a small release manifest that
conforms to `manifest.schema.json`.

Each file entry contains its destination-relative path, SHA-256 digest, size,
and the tiers that need it. URLs may be absolute; otherwise they are resolved
relative to the manifest URL. Test the public release in a clean directory:

```bash
pdeobs download --manifest https://example.org/pdeobs/v1/manifest.json \
  --tier tiny --output datasets/pdeobs-v1
pdeobs aggregate --input datasets/pdeobs-v1 --validate-shards \
  --output datasets/pdeobs-v1/summary.json
```

The release manifest should also record the dataset version, Git commit,
resolved generation-configuration hash, solver validation report, and license.
