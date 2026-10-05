# Parameters

All defaults are defined once in `frontveg.config.FrontVegConfig` and shared by the napari plugin and the command-line tools.

## Foreground (FrontVeg)

| Parameter | napari | CLI | Default | Effect |
|---|---|---|---|---|
| Sigma | Sigma (depth) | `--sigma` | 1.1 | Gaussian smoothing of the depth histogram. Higher values merge close depth modes. |
| Peak distance | Peak distance (depth) | `--peak_dist` | 8 | Minimal distance between two depth modes, in bins (1 to 20). |
| Peak height | | `--peak_height` | 0.5 | Minimal height of a mode, in pixels. |
| Smoothing | always on | `--smoothed` / `--no-smoothed` | on | Smooth the histogram before peak detection. |
| Encoder | | `--encoder` | `vitl` | Depth-Anything-V2 size. The installation guide downloads the `vitl` checkpoint; other sizes need their own checkpoint. |

## Instances (SAM3)

| Parameter | napari | CLI | Default | Effect |
|---|---|---|---|---|
| Prompt | Object to segment | `--prompt` | `leaf` | English noun describing the organ. |
| Tile overlap | Tile overlap (SAM3) | `--overlap` | 0.5 | Overlap between 640 px tiles, in `[0, 1)`. |
| Tile size | | `--tile_size` | 640 | Tile side in pixels. |
| Confidence | | `--conf_threshold` | 0.35 | Minimal SAM3 score of a detection. |

## Tuning guide

| Symptom | Try |
|---|---|
| Background rows are kept in the foreground | Increase **Peak distance** (10 to 14), or decrease **Sigma** |
| Parts of the front row are missing | Decrease **Peak distance** (5 to 6), or increase **Sigma** |
| Warning *"No depth valley found"* | The scene has no clear depth layers; check that the image is a side view with a visible background |
| Objects cut along straight lines | Increase **Tile overlap** (0.6 to 0.75) |
| Processing too slow | Decrease **Tile overlap** (0.3 to 0.4) |
| Small objects missed | Lower `--conf_threshold` (0.25 to 0.3) |
| False detections | Raise `--conf_threshold` (0.45 to 0.5), or use a more specific prompt (`grape bunch` instead of `fruit`) |

> [!TIP]
> Tune the foreground first with **Show FrontVeg foreground** checked, then the SAM3 parameters.

## Writing prompts

SAM3 prompts are short English noun phrases. Start generic (`leaf`, `fruit`), then refine if needed (`grape bunch`, `apricot`). The [examples gallery](index.md) shows leaves and fruits of vine, apricot and apple trees.
