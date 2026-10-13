// Copy BibTeX
const copyButton = document.querySelector("[data-copy-bib]");
const bibtex = document.querySelector("#bibtex");

if (copyButton && bibtex) {
  copyButton.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(bibtex.textContent.trim());
      const original = copyButton.textContent;
      copyButton.textContent = "Copied";
      window.setTimeout(() => {
        copyButton.textContent = original;
      }, 1400);
    } catch {
      copyButton.textContent = "Select text";
    }
  });
}

// Dataset tabs for the main imputation table
document.querySelectorAll(".tabs").forEach((list) => {
  const buttons = list.querySelectorAll("[data-panel]");
  buttons.forEach((btn) => {
    btn.addEventListener("click", () => {
      buttons.forEach((b) => {
        const on = b === btn;
        b.setAttribute("aria-selected", String(on));
        document.getElementById(b.dataset.panel).hidden = !on;
      });
    });
  });
});
