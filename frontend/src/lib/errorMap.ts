// Absolute error map |output - reference| between two images of the same size, computed in the
// browser from the PNGs the API returns. Inputs are RGBA pixel arrays (canvas ImageData.data).

export interface ErrorMap {
  /** RGBA pixels of a greyscale map: brighter = larger error, amplified by `gain`. */
  pixels: Uint8ClampedArray<ArrayBuffer>;
  /** Mean absolute error over all pixels and RGB channels, in [0, 1] (the L1 distance). */
  meanAbsError: number;
  /** Largest absolute error of any channel, in [0, 1]. */
  maxAbsError: number;
}

export function absoluteErrorMap(output: Uint8ClampedArray, reference: Uint8ClampedArray, gain = 4): ErrorMap {
  if (output.length !== reference.length || output.length % 4 !== 0) {
    throw new Error("absoluteErrorMap: images must have the same size");
  }
  const pixels = new Uint8ClampedArray(output.length);
  let sum = 0;
  let max = 0;
  for (let i = 0; i < output.length; i += 4) {
    const dr = Math.abs(output[i] - reference[i]);
    const dg = Math.abs(output[i + 1] - reference[i + 1]);
    const db = Math.abs(output[i + 2] - reference[i + 2]);
    sum += dr + dg + db;
    max = Math.max(max, dr, dg, db);
    const value = ((dr + dg + db) / 3) * gain; // Uint8ClampedArray clamps to 255
    pixels[i] = pixels[i + 1] = pixels[i + 2] = value;
    pixels[i + 3] = 255;
  }
  const channels = (output.length / 4) * 3;
  return { pixels, meanAbsError: channels ? sum / channels / 255 : 0, maxAbsError: max / 255 };
}
