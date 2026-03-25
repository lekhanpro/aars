const samples = [
  {
    id: "q_bm25",
    label: "BM25",
    question: "What sparse ranking algorithm rewards exact term overlap?",
    answer: "BM25.",
    badges: ["strategy: keyword", "type: factual", "answer target: exact term overlap"],
    docs: ["bm25"],
  },
  {
    id: "q_chroma",
    label: "Chroma",
    question: "Which component in AARS stores dense embeddings for semantic search?",
    answer: "ChromaDB.",
    badges: ["strategy: vector", "type: analytical", "answer target: dense storage"],
    docs: ["chroma"],
  },
  {
    id: "q_bert_transformer",
    label: "BERT",
    question: "Who introduced the architecture used by BERT?",
    answer: "Ashish Vaswani and colleagues.",
    badges: ["strategy: graph", "type: multi-hop", "answer target: linked architecture history"],
    docs: ["transformer_history", "bert_architecture"],
  },
  {
    id: "q_openai_city",
    label: "HQ city",
    question: "In which city is the company led by Sam Altman headquartered?",
    answer: "San Francisco.",
    badges: ["strategy: graph", "type: multi-hop", "answer target: CEO -> company -> city"],
    docs: ["openai_ceo", "openai_hq"],
  },
  {
    id: "q_rrf",
    label: "Fusion",
    question: "Which method merges ranked lists before MMR reranking?",
    answer: "Reciprocal Rank Fusion, or RRF.",
    badges: ["strategy: hybrid", "type: analytical", "answer target: fusion step"],
    docs: ["rrf", "mmr"],
  },
];

function renderSampleButtons() {
  const mount = document.querySelector("[data-sample-list]");
  if (!mount) {
    return;
  }

  samples.forEach((sample, index) => {
    const button = document.createElement("button");
    button.className = "sample-chip";
    button.type = "button";
    button.textContent = sample.label;
    button.dataset.sampleId = sample.id;
    button.addEventListener("click", () => selectSample(sample.id));
    mount.appendChild(button);
    if (index === 0) {
      button.classList.add("is-active");
    }
  });
}

function selectSample(id) {
  const sample = samples.find((item) => item.id === id) || samples[0];
  document.querySelectorAll(".sample-chip").forEach((button) => {
    button.classList.toggle("is-active", button.dataset.sampleId === sample.id);
  });

  const sampleId = document.querySelector("[data-sample-display-id]");
  const sampleQuestion = document.querySelector("[data-sample-question]");
  const sampleAnswer = document.querySelector("[data-sample-answer]");
  const sampleBadges = document.querySelector("[data-sample-badges]");
  const sampleDocs = document.querySelector("[data-sample-docs]");

  if (!sampleId || !sampleQuestion || !sampleAnswer || !sampleBadges || !sampleDocs) {
    return;
  }

  sampleId.textContent = sample.id;
  sampleQuestion.textContent = sample.question;
  sampleAnswer.textContent = sample.answer;

  sampleBadges.innerHTML = "";
  sample.badges.forEach((badgeText) => {
    const badge = document.createElement("span");
    badge.className = "sample-badge";
    badge.textContent = badgeText;
    sampleBadges.appendChild(badge);
  });

  sampleDocs.innerHTML = "";
  sample.docs.forEach((docId) => {
    const item = document.createElement("li");
    item.textContent = docId;
    sampleDocs.appendChild(item);
  });
}

function setupCopyButtons() {
  document.querySelectorAll("[data-copy-target]").forEach((button) => {
    button.addEventListener("click", async () => {
      const targetId = button.getAttribute("data-copy-target");
      const target = targetId ? document.getElementById(targetId) : null;
      if (!target) {
        return;
      }

      try {
        await navigator.clipboard.writeText(target.textContent || "");
        button.classList.add("is-copied");
        button.textContent = "Copied";
        window.setTimeout(() => {
          button.classList.remove("is-copied");
          button.textContent = "Copy";
        }, 1400);
      } catch (error) {
        button.textContent = "Copy failed";
        window.setTimeout(() => {
          button.textContent = "Copy";
        }, 1400);
      }
    });
  });
}

function setupMobileNav() {
  const toggle = document.querySelector("[data-menu-toggle]");
  const nav = document.querySelector("[data-nav]");
  if (!toggle || !nav) {
    return;
  }

  toggle.addEventListener("click", () => {
    nav.classList.toggle("is-open");
  });
}

renderSampleButtons();
selectSample(samples[0].id);
setupCopyButtons();
setupMobileNav();
