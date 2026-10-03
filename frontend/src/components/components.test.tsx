import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { ApiError } from "../api/client";
import { BRANCH_LABEL, CLASS_LABEL } from "../lib/format";
import { PNG } from "../test/fixtures";
import { DownloadButton } from "./DetailsCard";
import { RequestError } from "./RequestError";
import { ScoreBars } from "./ScoreBars";

describe("ScoreBars", () => {
  const scores = { clean: 0.02, salt: 0.05, blur: 0.89, occlusion: 0.04 };

  it("renders one bar per class with its percentage and width", () => {
    render(
      <ScoreBars label="Classifier probabilities" variant="grid" scores={scores} labels={CLASS_LABEL} highlight="blur" highlightTag="Predicted" />,
    );
    const bars = screen.getAllByRole("progressbar");
    expect(bars).toHaveLength(4);
    const blur = screen.getByRole("progressbar", { name: "Blur" });
    expect(blur).toHaveAttribute("aria-valuenow", "89");
    expect(blur).toHaveAttribute("aria-valuetext", "89.0%");
    expect(blur.firstElementChild).toHaveStyle({ width: "89%" });
    expect(screen.getByText("2.0%")).toBeInTheDocument();
  });

  it("highlights the given class and marks a second one", () => {
    render(
      <ScoreBars
        label="Expert weights"
        variant="list"
        scores={scores}
        labels={BRANCH_LABEL}
        highlight="blur"
        highlightTag="Main contributor"
        marker={{ cls: "salt", tag: "True class" }}
      />,
    );
    const group = screen.getByRole("group", { name: "Expert weights" });
    const tag = within(group).getByText("Main contributor");
    expect(tag.parentElement).toHaveTextContent("Blur expert");
    expect(within(group).getByText("True class").parentElement).toHaveTextContent("Salt & Pepper expert");
  });

  it("clamps values outside [0, 1] for the bar width", () => {
    render(
      <ScoreBars label="x" variant="list" scores={{ ...scores, blur: 1.2 }} labels={CLASS_LABEL} highlight="blur" highlightTag="t" />,
    );
    expect(screen.getByRole("progressbar", { name: "Blur" }).firstElementChild).toHaveStyle({ width: "100%" });
  });
});

describe("DownloadButton", () => {
  it("links the PNG data URL with a download file name", () => {
    render(<DownloadButton href={PNG} filename="cat_restored-universal.png" label="Download result" />);
    const link = screen.getByRole("link", { name: "Download result" });
    expect(link).toHaveAttribute("href", PNG);
    expect(link).toHaveAttribute("download", "cat_restored-universal.png");
  });

  it("is disabled without a result", () => {
    render(<DownloadButton href={null} filename="x.png" label="Download sketch" />);
    expect(screen.getByRole("button", { name: "Download sketch" })).toBeDisabled();
    expect(screen.queryByRole("link")).toBeNull();
  });
});

describe("RequestError", () => {
  it("shows the backend detail with a title for the status", () => {
    render(<RequestError error={new ApiError(503, "Model file moe.onnx is missing from /models.")} />);
    const alert = screen.getByRole("alert");
    expect(alert).toHaveTextContent("Model not available (503)");
    expect(alert).toHaveTextContent("Model file moe.onnx is missing from /models.");
  });

  it("renders nothing without an error", () => {
    const { container } = render(<RequestError error={null} />);
    expect(container).toBeEmptyDOMElement();
  });
});
