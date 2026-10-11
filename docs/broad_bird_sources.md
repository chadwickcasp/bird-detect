# Broader bird imagery and San Francisco sources

Checked 2026-10-10. Revised priorities after the request for broader collections: general labeled bird imagery plus geographically filtered local photos. Earlier camera-specific findings remain in [mobile_bird_data_search.md](mobile_bird_data_search.md).

## Recommended sources

| Source | Why use it | Limitation |
|---|---|---|
| [Open Images Bird detection gallery](https://storage.googleapis.com/openimages/web/visualizer/index.html?type=detection&set=train&c=%2Fm%2F015p6) | General bird class, varied photographers and settings, existing boxes | Must select actual flight photos and exclude depictions. Camera model is not established by the class label. |
| [GBIF San Francisco-area photo gallery](https://www.gbif.org/occurrence/gallery?taxon_key=212&media_type=StillImage&has_coordinate=true&has_geospatial_issue=false&geometry=POLYGON%28%28-122.52%2037.70%2C-122.35%2037.70%2C-122.35%2037.84%2C-122.52%2037.84%2C-122.52%2037.70%29%29) | Local occurrence taxonomy, dates, and links to original photos | No detector boxes verified; local sightings include all poses. Camera and flight status require image-level checks. |
| [iNaturalist 2017 competition](https://github.com/visipedia/inat_comp/blob/master/2017/README.md) | Broad community collection: 964 bird species, 214,295 training bird images; 283,931 training and 17,314 validation bird boxes | Noncommercial research/education terms; images predate the Pixel 9 era and are resized to a maximum dimension of 800 pixels. |

Open Images annotations are CC BY 4.0; images are listed as CC BY 2.0 with individual-license verification advised by the publisher. [Official description](https://storage.googleapis.com/openimages/web/factsfigures_v7.html).

## Visually checked Open Images examples

- [Falcon 5 — 2a9924cc769904c2](https://storage.googleapis.com/openimages/web/visualizer/index.html?type=all&set=train&c=%2Fm%2F015p6&id=2a9924cc769904c2): flying raptor against sky, with a visible existing bounding box after masks, points, captions, and traces were disabled. Camera model and location unverified.
- [Bridgeport scene — 6aaea3ee5b0c2f02](https://storage.googleapis.com/openimages/web/visualizer/index.html?type=all&set=train&c=%2Fm%2F015p6&id=6aaea3ee5b0c2f02): small bird in flight near a lighthouse at dusk, but no bird box was visible in the inspected overlay. Do not treat the positive Bird image label as a detector annotation.
- [Bird against a blue sky — 96997fa30b66a786](https://storage.googleapis.com/openimages/web/visualizer/index.html?type=all&set=train&c=%2Fm%2F015p6&id=96997fa30b66a786): boxed gull, visibly perched. The title alone does not establish flight.

These examples show why selection must inspect both the image and its annotations rather than accepting search titles or image-level labels.

## Completed GBIF query

An unauthenticated public occurrence search returned **95,493 occurrence records with still images** for Aves in the rectangle west/east −122.52/−122.35, south/north 37.70/37.84. Coordinates were required and records flagged with geospatial issues were excluded. The rectangle approximates the city area, including adjacent water; it is not the administrative boundary and excludes the Farallon Islands.

This is a count of observation records, not unique photos, flying-bird photos, smartphone photos, or species abundance. The query had no date or license restriction. Retrieved example records included:

| Species | Date | Original observation |
|---|---|---|
| Lophodytes cucullatus | 2026-01-02 | [333271700](https://www.inaturalist.org/observations/333271700) |
| Sturnella neglecta | 2026-01-03 | [333372244](https://www.inaturalist.org/observations/333372244) |
| Bucephala albeola | 2026-01-03 | [333381775](https://www.inaturalist.org/observations/333381775) |
| Calidris alba | 2026-01-03 | [333634157](https://www.inaturalist.org/observations/333634157) |
| Zonotrichia leucophrys | 2026-01-01 | [333117092](https://www.inaturalist.org/observations/333117092) |

The returned photos in these five records were CC BY-NC 4.0. Their flight pose, camera model, and image quality were not visually audited. They are examples of source metadata, not selected training images.

GBIF links media hosted by publishers. Its [image-cache documentation](https://techdocs.gbif.org/en/openapi/images) says cached images are limited to 1200 × 1200 with JPEG quality restrictions; source originals are preferable when evaluating real camera artifacts. The multimedia license can differ from the occurrence-data license.

## Selection for this project

Use Open Images as a general labeled base and GBIF/iNaturalist as a local supplement. Keep one generic Bird detection class; species metadata can guide coverage without requiring a species classifier.

For flight examples, select clear sky, cloud/fog backgrounds, coast/urban clutter, flocks, tiny silhouettes, and varied wing positions. Include naturally blurry, poorly focused, zoomed, compressed, and low-light images when available. These conditions are selection targets, not verified coverage claims for the full collections. None of the broad sources above establishes Pixel-specific processing realism.

For local sampling, balance contributors, observation dates, habitats, and bird sizes. Avoid treating the most frequently photographed species as the true local abundance distribution. Keep photos from the same observation/session in one split.

A follow-up query for recent CC0/CC BY candidates was not executed: automatic approval review failed because its service hit a usage limit. No recent-license-filtered candidate manifest or image download was produced.
