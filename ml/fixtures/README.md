# Parity fixtures

These three images check that the phone decoder matches the host reference. They are not training or evaluation data.

| File | What it checks | Source |
| --- | --- | --- |
| `bird_gull.jpg` | One bird in flight, landscape frame | [Ring-billed gull in flight (94615)](https://commons.wikimedia.org/wiki/File:Ring-billed_gull_in_flight_(94615).jpg) by Rhododendrites, [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0). Downloaded as the 1280 px Commons thumbnail. |
| `bird_gull_rot90.jpg` | Portrait frame and letterbox padding on the other axis | `bird_gull.jpg` rotated 90 degrees clockwise in pixels. Same author and license. It carries no EXIF orientation tag. |
| `empty_sky.jpg` | No-bird negative | [Cloudy blue sky 5029](https://commons.wikimedia.org/wiki/File:Cloudy_blue_sky_5029.jpg) by Yassine fessili, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0). Downloaded as the 1280 px Commons thumbnail. |

The images were re-encoded as JPEG quality 92 without EXIF. The Android app bundles this folder as assets for its fixture run.
