// Entry point. Fonts and icons are bundled from npm packages (no Google Fonts / CDN requests), so
// the app also works without internet access.
import "@fontsource-variable/inter";
import "material-symbols/outlined.css";
import "./index.css";

import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
