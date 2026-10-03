// State of the restoration "Input" card shared by the Universal, Hard-Routed and Soft MoE
// workspaces: the image, the corruption (type, level or custom parameters, seed) and the request
// that is sent for it.
import { useState } from "react";
import type { CorruptionChoice, CorruptionRequest } from "../../api/client";
import type { Level, OptionsResponse } from "../../api/types";
import { useSelectedImage } from "../../hooks/useSelectedImage";

export const MAX_SEED = 2 ** 31 - 1;

/** null for an empty (random) seed, a number, or an error message. */
export function parseSeed(text: string): { seed: number | null; error: string | null } {
  const trimmed = text.trim();
  if (trimmed === "") return { seed: null, error: null };
  if (!/^\d+$/.test(trimmed) || Number(trimmed) > MAX_SEED) {
    return { seed: null, error: `The seed must be a whole number from 0 to ${MAX_SEED}.` };
  }
  return { seed: Number(trimmed), error: null };
}

/** Parameters of a fixed level from GET /api/options, e.g. {k: 5, sigma: 1.5} for blur/medium. */
export function levelParams(
  options: OptionsResponse | null,
  type: CorruptionChoice,
  level: Level,
): Record<string, number> | null {
  if (!options || type === "none") return null;
  return options.corruptions.find((c) => c.type === type)?.levels[level] ?? null;
}

export function useRestorationForm() {
  const image = useSelectedImage();
  // The Stitch design opens with Blur / Medium selected.
  const [choice, setChoice] = useState<CorruptionChoice>("blur");
  const [level, setLevel] = useState<Level>("medium");
  const [customEnabled, setCustomEnabled] = useState(false);
  const [customValues, setCustomValues] = useState<Partial<Record<CorruptionChoice, Record<string, number>>>>({});
  const [seedText, setSeedText] = useState("");

  const seed = parseSeed(seedText);
  const usesCorruption = choice !== "none";

  /** Custom values of the chosen type, starting from the parameters of the chosen level. */
  const customFor = (options: OptionsResponse | null): Record<string, number> | null =>
    customValues[choice] ?? levelParams(options, choice, level);

  const setCustomValue = (options: OptionsResponse | null, name: string, value: number) =>
    setCustomValues((all) => ({ ...all, [choice]: { ...(customFor(options) ?? {}), [name]: value } }));

  /** The corruption part of the request, or null while the seed is invalid. */
  const buildRequest = (options: OptionsResponse | null): CorruptionRequest | null => {
    if (seed.error) return null;
    const custom = usesCorruption && customEnabled ? customFor(options) : null;
    return { choice, level, custom, seed: seed.seed };
  };

  return {
    image,
    choice,
    setChoice,
    level,
    setLevel,
    customEnabled,
    setCustomEnabled,
    customFor,
    setCustomValue,
    seedText,
    setSeedText,
    seedError: seed.error,
    buildRequest,
  };
}

export type RestorationForm = ReturnType<typeof useRestorationForm>;
