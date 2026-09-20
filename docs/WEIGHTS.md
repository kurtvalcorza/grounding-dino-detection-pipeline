# Weight provenance and DIMER hosting

- Upstream: `IDEA-Research/grounding-dino-tiny`
- Immutable revision: `a2bb814dd30d776dcf7e30523b00659f4f141c71`
- Weight format: SafeTensors (`model.safetensors`, 689,359,096 bytes)
- Manifest: `weights/grounding-dino-tiny/dimer-base-manifest.json` (9 files including the BERT tokenizer assets, 690,308,125 bytes total, per-file SHA-256)
- Upstream weight license: Apache-2.0
- DIMER hosting: Apache-2.0 permits use, modification, distribution, and commercial use subject to preservation of the license and notices. The Git repository does not vendor the checkpoint (`weights/**/*.safetensors` is git-ignored); DIMER may mirror the pinned snapshot in its model store under the upstream license.
- Fresh clone: `stage_missing_files(allow_download=True)` fetches only the manifest-listed files absent on disk, at the pinned revision, into the snapshot directory; `verify_snapshot()` then checks every file before any load.
- Loader trust boundary: Transformers `AutoModelForZeroShotObjectDetection` / `AutoProcessor` with `trust_remote_code=False`, `local_files_only=True` from the verified directory.
- Adaptation sample cache: `weights/bccd/` (git-ignored) holds the 364 pinned BCCD photographs (`BloodImage_*.jpg`, about 7.6 MB) fetched by `samples.fetch_corpus` from `https://raw.githubusercontent.com/Shenggan/BCCD_Dataset/d272fb14cdff6e473fafeeeba32aba5f560e9e43/BCCD/JPEGImages/` (MIT); every cached file is re-hashed against `SAMPLE_IMAGES` on read. Adapters written by `save_artifact` (`adapter.safetensors` + `manifest.json`; the decoder and the box / contrastive heads only) are outputs, not part of the snapshot, and `from_artifact` re-verifies the snapshot before overlaying them.
