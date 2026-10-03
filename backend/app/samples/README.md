# Bundled sample images

Clean images offered in the app's sample pickers, served by `GET /api/samples` and usable in
every POST endpoint as `sample_id` instead of an upload.

- `pets/` — Oxford-IIIT Pet images for the restoration workspaces (Tasks 1-3). Use images from
  the official **test** split so the demo shows unseen data.
- `faces/` — face photos for the Face-to-Sketch Generator (Task 4), e.g. FS2K test photos.

Rules:
- Formats: `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`; other files (like this README) are ignored.
- The id is `<folder>-<file name without extension>`, lower-cased, with characters other than
  letters, digits and `_` replaced by `-` (e.g. `pets/Abyssinian_12.jpg` -> `pets-abyssinian_12`).
- Keep them small (a few hundred pixels, < 200 KB each): they are copied into the Docker image.
- The folder is scanned when the backend starts; restart it after adding images.
- Check the dataset licences before redistributing images in the repository.

## Attribution

`pets/` holds eight images from the **official test split** of the Oxford-IIIT Pet Dataset
(Abyssinian_201, Bengal_192, Persian_21, Siamese_209, beagle_195, german_shorthaired_191,
pomeranian_191, samoyed_191), downscaled to at most 384 px. O. M. Parkhi, A. Vedaldi,
A. Zisserman and C. V. Jawahar, "Cats and Dogs", CVPR 2012 —
<https://www.robots.ox.ac.uk/~vgg/data/pets/>, licensed CC BY-SA 4.0.

`faces/` is intentionally empty: FS2K's photos are not redistributed here. The Face-to-Sketch
Generator works with uploads and the webcam.
