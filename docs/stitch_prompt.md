# Google Stitch prompts

The application interface is designed in Google Stitch (stitch.withgoogle.com) before
implementation, as required by the assignment. Generate Prompt 1 first, then add the
other screens to the same project so they share the same design. Workspace names match
the assignment exactly.

## Prompt 1 — app shell + Universal Restoration

```
Design a simple, clean desktop web app called "GenAI Studio" that demonstrates four image AI models. Keep it minimal and lightweight: light theme, white cards on a very light gray background, one accent color (indigo), rounded corners, thin borders, no gradients, no heavy shadows, no illustrations, Inter font, plenty of whitespace. On mobile the columns stack vertically.

Top bar: app name "GenAI Studio" on the left; four tabs: "Universal Restoration", "Hard-Routed Restoration", "Soft Mixture-of-Experts Restoration", "Face-to-Sketch Generator"; on the right a small status pill "Backend online" with a green dot.

This screen is the "Universal Restoration" workspace (first tab active). Two columns:

Left column (narrow), one card titled "Input":
- an upload dropzone: "Drop an image or click to upload"
- "Or pick a sample": a row of 6 small square pet photo thumbnails
- "Corruption": segmented control with None / Salt & Pepper / Blur / Occlusion
- "Severity": segmented control with Low / Medium / High
- a full-width primary button "Restore"

Right column (wide):
- three square image panels in a row with captions "Original", "Model input", "Restored output"
- below them a compact details card with label-value rows: Corruption "Gaussian blur · kernel 5 · σ 1.5", Model "udae.onnx", Inference time "12.4 ms", Image size "128 × 128"
- a secondary button "Download result"
```

## Prompt 2 — Hard-Routed Restoration

```
Using exactly the same design, create the "Hard-Routed Restoration" screen. Same top bar with this tab active and the same left "Input" card, plus a small toggle "Routing: Predicted / Oracle" above the Restore button.

Right column: the same three image panels ("Original", "Model input", "Restored output"), then a card "Classifier" with four thin horizontal probability bars with percentages: Clean, Salt & Pepper, Blur, Occlusion (the highest bar in the accent color), and two badges: "Predicted: Blur" and "Expert used: Blur specialist". Then the compact details card with Inference time split into Classifier, Expert and Total, and the "Download result" button.
```

## Prompt 3 — Soft Mixture-of-Experts Restoration

```
Using exactly the same design, create the "Soft Mixture-of-Experts Restoration" screen. Same top bar with this tab active and the same left "Input" card.

Right column: the same three image panels, then a card "Expert weights" with four thin horizontal bars with percentages: Identity (clean), Salt & Pepper expert, Blur expert, Occlusion expert; the largest bar is in the accent color with a small label "Main contributor". Under the bars, a row of four small thumbnails showing each expert's output with its weight underneath. Then the compact details card (Model "moe.onnx", Inference time) and the "Download result" button.
```

## Prompt 4 — Face-to-Sketch Generator

```
Using exactly the same design, create the "Face-to-Sketch Generator" screen. Same top bar with this tab active.

Left card titled "Photo": an upload dropzone, a secondary button "Use webcam" with a camera icon, a "Sketch style" selector with three options "Style 1", "Style 2", "Style 3", and a full-width primary button "Generate sketch".

Right column: two large square panels side by side, "Photo" and "Generated sketch" (a grayscale pencil sketch of the face), a compact details card (Style "Style 2", Model "generator.onnx", Inference time "35.2 ms") and a primary button "Download sketch".
```

## Evidence for the report

- Screenshot of the Stitch canvas showing all four screens, plus one screenshot per screen.
- These prompts (this file) as the record of what was asked.
- If Stitch offers a code export (HTML with Tailwind classes), save each screen's export in
  `design/stitch/` in the repository; the React frontend is built from it.
