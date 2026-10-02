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
