// ------------------------------------------------------------------
//  Single source of truth for the landing copy.
//  Numbers correspond to the real model where known; measurement
//  fields are filled after training (marked `measured: false`).
// ------------------------------------------------------------------

export const PARAMS = 47_640_968;
export const ACTIVE = 38_793_608;
export const HEADROOM = 2_359_032;
export const VOCAB = 24_000;
export const EXPERTS_N = 8;
export const TOP_K = 2;
export const D_MODEL = 384;
export const BUDGET = 50_000_000;
export const TOKENS_TARGET = "2–4B";
export const SEQ_LEN = 1024;

export type StackNode = {
  id: string;
  name: string;
  sub: string;
  kind: "data" | "block" | "gate" | "moe" | "head";
};

// Interleaved Mamba ∥ attention stack, as laid out in architecture.py
export const STACK: StackNode[] = [
  { id: "emb", name: "EMBED", sub: "token lookup · tied head", kind: "data" },
  { id: "m0", name: "MB ∥ AT", sub: "mamba · attention", kind: "block" },
  { id: "m1", name: "MB ∥ AT", sub: "mamba · attention", kind: "block" },
  { id: "m2", name: "MB ∥ AT", sub: "mamba · attention", kind: "block" },
  { id: "m3", name: "MB ∥ AT", sub: "mamba · attention", kind: "block" },
  { id: "m4", name: "MB ∥ AT", sub: "mamba · attention", kind: "block" },
  { id: "m5", name: "MB ∥ AT", sub: "mamba · attention", kind: "block" },
  { id: "m6", name: "MB ∥ AT", sub: "mamba · attention", kind: "block" },
  { id: "gate", name: "GATE", sub: "pre-MoE norm", kind: "gate" },
  { id: "moe", name: "MOE ×8", sub: "top-2 · domain-seeded", kind: "moe" },
  { id: "head", name: "HEAD", sub: "lm_head · shared embed", kind: "head" },
];

// Domain taxonomy mirrors src/data/domains.py ↔ src/model/moe.py
export type Domain = {
  id: string;
  expert: string;
  short: string;
  css: string;
  desc: string;
};

export const DOMAINS: Domain[] = [
  { id: "math", expert: "E00", short: "MATH", css: "domain-math", desc: "gsm8k · the_pile · arithmetic CoT" },
  { id: "code", expert: "E01", short: "CODE", css: "domain-code", desc: "code corpora · structured syntax" },
  { id: "logic", expert: "E02", short: "LOGIC", css: "domain-logic", desc: "logicqa · chain-of-thought" },
  { id: "common", expert: "E03", short: "COMMON", css: "domain-common", desc: "hellaswag · arc · winogrande · piqa" },
  { id: "sql", expert: "E04", short: "SQL", css: "domain-sql", desc: "spider · schemas · queries" },
  { id: "json", expert: "E05", short: "JSON", css: "domain-json", desc: "json instructions · objects" },
  { id: "g1", expert: "E06", short: "GEN·1", css: "domain-general1", desc: "fineweb · general web text" },
  { id: "g2", expert: "E07", short: "GEN·2", css: "domain-general2", desc: "fineweb-edu · educational text" },
];

export type BenchRow = {
  id: string;
  task: string;
  metric: string;
  ours: { value: string | null; measured: boolean };
  gpt2: string | null;
  pythia: string | null;
  note: string;
};

// ours.value filled after training → re-render automatically.
export const BENCHMARKS: BenchRow[] = [
  { id: "hellaswag", task: "HELLASWAG", metric: "acc_norm", ours: { value: null, measured: false }, gpt2: "32.7", pythia: "27–30", note: "common-sense inference, 4-way MC" },
  { id: "arc-e", task: "ARC-EASY", metric: "acc_norm", ours: { value: null, measured: false }, gpt2: "43.3", pythia: "40–45", note: "grade-school science, 4-way MC" },
  { id: "piqa", task: "PIQA", metric: "acc_norm", ours: { value: null, measured: false }, gpt2: "64.2", pythia: "61–63", note: "physical commonsense, 2-way MC" },
  { id: "wino", task: "WINOGRAANDE", metric: "acc", ours: { value: null, measured: false }, gpt2: "49.9", pythia: "50–52", note: "pronoun resolution, 2-way MC" },
  { id: "wiki", task: "WIKITEXT-103", metric: "word_ppl", ours: { value: null, measured: false }, gpt2: "~38", pythia: "~44", note: "word-level perplexity ↓ lower wins" },
];

export type Principle = {
  n: string;
  title: string;
  body: string;
  stat: { value: string; label: string };
};

export const PRINCIPLES: Principle[] = [
  { n: "01", title: "Zero pretrained weights.", body: "Every parameter is randomized and learned on our own corpus. No distillation, no warm-start — the honest way to test a small architecture.", stat: { value: "0", label: "weights reused" } },
  { n: "02", title: "The budget is enforced, not hoped for.", body: "A two-gate pipeline checks the parameter math at build time — a static projection and a real-model count — and fails the run if the 50M line is crossed.", stat: { value: "50M", label: "hard ceiling, CI-enforced" } },
  { n: "03", title: "Routed by domain, seeded by curriculum.", body: "Eight experts, two active per token. A guide-loss teaches each expert its specialty over the first 2k steps, then anneals away.", stat: { value: "8→2", label: "experts per token" } },
  { n: "04", title: "One consumer GPU.", body: "fp16 training on a single notebook T4. If your architecture doesn't learn under that constraint, it isn't honest.", stat: { value: "1×T4", label: "fp16 AMP" } },
  { n: "05", title: "Structured output is a protocol.", body: "JSON / SQL / CoT wrapped in explicit special tokens — machine-parseable answers, extractable reasoning, zero templating hacks.", stat: { value: "6", label: "special tokens" } },
];

// Interactive mock demo — responses for the concept renderer.
export type MockEntry = { domain: string; prompt: string; response: string; experts: number[] };
export const MOCK_LIVE: MockEntry[] = [
  {
    domain: "json",
    prompt: "Return a JSON object with id 42, name Alice, age 29, city Kyiv.",
    response: "{ \"id\": 42, \"name\": \"Alice\", \"age\": 29, \"city\": \"Kyiv\" }",
    experts: [5, 3],
  },
  {
    domain: "sql",
    prompt: "Schema users(id,name,age,city). Query rows where city = 'Kyiv'.",
    response: "SELECT name FROM users WHERE city = 'Kyiv';",
    experts: [4, 6],
  },
  {
    domain: "cot",
    prompt: "A shop starts with 12 items and gets 7 more. Total?",
    response: "12 + 7 = 19 items.",
    experts: [0, 2],
  },
];

export type Milestone = { tag: string; title: string; body: string; status: "done" | "next" | "open" };
export const ROADMAP: Milestone[] = [
  { tag: "W1", title: "Architecture & the 50M gate", body: "Block accounting, dual parameter gates, attention core.", status: "done" },
  { tag: "W2", title: "Mamba & Mixture of Experts", body: "Selective SSM, top-2 router, guide curriculum.", status: "done" },
  { tag: "W3", title: "Data & the 24k tokenizer", body: "ByteLevel BPE, packing, memmap shards, domains.", status: "done" },
  { tag: "W4", title: "Training loop", body: "fp16 AMP, cosine LR, checkpointing, W&B.", status: "next" },
  { tag: "W5", title: "Structured SFT", body: "JSON / SQL / CoT protocol on top of the pretrained stack.", status: "open" },
  { tag: "W6", title: "Evaluation & demo", body: "Five mandatory benchmarks, web demo, submission.", status: "open" },
];

export const NAV = [
  { label: "017·MODEL", href: "#model" },
  { label: "02·ROUTER", href: "#router" },
  { label: "03·BENCH", href: "#bench" },
  { label: "04·LIVE", href: "#live" },
  { label: "05·BUILD", href: "#build" },
];

export const GITHUB = "https://github.com/qtttyr/MiraLM";
export const WANDB = "https://wandb.ai";