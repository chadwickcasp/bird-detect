# Mobile-camera bird imagery search — 2026-10-10

The strongest verified sky/phone match is the Pixel 8 Pro flock photograph below. No public collection was verified to combine recent-phone provenance, flying birds against sky, bounding boxes, and documented coverage of mobile-camera artifacts. Dataset descriptions and preview images were checked; archives were not downloaded or annotation quality audited.

## Verified phone sources

### Pixel 8 Pro: starlings over Worthing Pier

[Source and original download](https://commons.wikimedia.org/wiki/File:Starlings_flying_across_Worthing_Pier_2026-07-02.jpg)

- Photographer: Andy Li; CC0 1.0.
- EXIF: Google Pixel 8 Pro, 2 July 2026; 8160 × 6144; 18 mm, f/2.8, ISO 18, about 1/357 s, 110 mm equivalent. Software identifies HDR+ processing.
- Preview visually confirms many small flying birds against sky, with pier/sea clutter below. No bounding-box annotations supplied.
- Best use: real Pixel still-photo example to annotate, including small birds and flocks. Low ISO does not establish high-noise coverage. Original-resolution artifact measurements have not been made.

The Pixel 8 Pro and Pixel 9 Pro both have 48 MP, f/2.8, 5× telephoto cameras, making this a useful hardware-relative substitute. It is not a direct substitute for a base Pixel 9 main-camera view, nor evidence of identical processing. [Google specifications](https://support.google.com/pixelphone/answer/7158570?hl=en).

### iPhone 14 Pro: Hiroshima heron flight sequence

[Sequence listing](https://commons.wikimedia.org/wiki/Category:Ardea_cinerea_in_Japan) contains files named “Ardea cinerea in flight in Hiroshima” 01–17.

Checked metadata/license pages: [01](https://commons.wikimedia.org/wiki/File:Ardea_cinerea_in_flight_in_Hiroshima_01.jpg), [06](https://commons.wikimedia.org/wiki/File:Ardea_cinerea_in_flight_in_Hiroshima_06.jpg), [17](https://commons.wikimedia.org/wiki/File:Ardea_cinerea_in_flight_in_Hiroshima_17.jpg).

- Photographer: ノボホショコロトソ; checked files are CC BY 4.0.
- Checked files: iPhone 14 Pro, 4032 × 3024, 12 August 2026, 9 mm f/2.8, ISO 32; capture times span 11:28:56–11:29:07.
- Files 01 and 06 record approximately 2.60× digital zoom and 201 mm equivalent field of view.
- Frame 06 visually shows a flying heron against water, so this is a phone/zoom supplement rather than the strongest sky match. No boxes supplied. Other sequence images were not all visually audited.
- Keep the entire sequence in one data split because its images share the bird, setting, and capture session.

## Sources with existing bounding boxes

| Source | What is documented | Fit and limitation |
|---|---|---|
| [IBERBIRDS](https://zenodo.org/records/15090760) | 4,000 flying-bird photos, 10 species, YOLO labels; 4.6 GB ZIP; record lists CC BY 4.0 | Strong new labeled starting point. Images gathered from ornithology platforms, not verified phone captures. Curated single-bird images do not replace flock examples. |
| [BirdMOT detection dataset](https://github.com/fids-openresearchlab/BirdMOT-yolov3-det) | 3,811 3840 × 2160 images, 10,044 annotated birds, 42 hard negatives; authors describe tiny birds and motion blur | Useful difficult examples. Camera model and dataset reuse license were not established from the README. Low-light data is listed as future work, not verified coverage. |
| [FBD-SV-2024](https://github.com/Ziwei89/FBD-SV-2024_github) | 483 clips, 28,694 frames; bounding boxes and tracking information; surveillance capture | Existing project candidate, not phone footage. [Download record](https://figshare.com/s/1ca0193680f894a65371) has research/noncommercial language despite CC BY metadata; do not assume unrestricted reuse. |
| [AirBirds](https://airbirdsdata.github.io/) | 118,312 1080p images, 409,967 boxes, often sub-10-pixel birds; varied seasons/weather | Useful distant-bird coverage, not phone-provenance evidence. Approximately 165 GB, access form, CC BY-NC-SA 4.0. |

IBERBIRDS direct archive: [IBERBIRDS_dataset.zip](https://zenodo.org/records/15090760/files/IBERBIRDS_dataset.zip?download=1).

## Camera artifacts: remaining coverage to collect

Recommended coverage categories, not claims that all occur in the sources above:

- Small birds measured after the detector's actual resize, including uncertain distant silhouettes.
- Wing/camera motion blur, missed focus, and panning/stabilization effects.
- Dusk/high-ISO noise and denoising loss of detail.
- Digital zoom, sharpening, resizing, and image/video compression.
- Backlighting, bright clouds, exposure changes, and multi-frame processing behavior.
- Empty skies and confusing non-birds such as insects, aircraft, and distant debris.

Google documents the motion, alignment, noise, and exposure tradeoffs involved in [HDR+ bracketing](https://research.google/blog/hdr-with-bracketing-on-pixel-phones/) and [Super Res Zoom](https://research.google/blog/see-better-and-further-with-super-res-zoom-on-the-pixel-3/). This supports collecting those conditions; it does not prove a particular defect exists in any candidate photograph.

For a live Android detector, collect a separate evaluation set from the actual app camera pipeline. CameraX ImageAnalysis commonly supplies YUV frames; saved HDR+ stills should not be treated as equivalent inputs. [CameraX documentation](https://developer.android.com/media/camera/camerax/analyze). Split by capture session, not neighboring frames. Synthetic blur/noise/compression can supplement training but cannot establish Pixel-specific realism.

## Excluded or reference-only leads

- A Pixel 9 Pro XL Unsplash image titled [“A bird flying through a cloud filled sky”](https://unsplash.com/photos/a-bird-flying-through-a-cloud-filled-sky-YQIKZvvMS3E) did not show an identifiable bird in the inspected preview. Its title is insufficient evidence.
- [This iPhone 15 Pro image](https://unsplash.com/photos/a-flock-of-birds-flying-through-a-cloudy-sky-f8sRq_BRMx8) visually contains four birds against cloudy sky, but [Unsplash terms, section 8](https://unsplash.com/terms) restrict ML/AI dataset uses. Treat it as a visual reference unless separately licensed for the intended use.

Recommended starting combination: IBERBIRDS for labeled flight examples, the Commons phone images for a small manually annotated phone-specific supplement, and new app-pipeline captures for a meaningful Pixel evaluation set.
