//! Benchmark: needle-retention eval over 50 synthetic long conversations.
//!
//! Each conversation opens with an early-turn fact (a constraint the user
//! states once), buries it under filler chatter, and ends with a question
//! that can only be answered if the fact survived compaction.
//!
//! What this measures is RETENTION — did the compacted history still contain
//! the answer keyword — not model answers. No model runs here; retention is
//! the necessary precondition for a correct answer and the part CTX controls.
//! Any table quoting this eval must say "retained", never "answered".
//!
//! Hardening (see benchmarks/README.md for the full methods note):
//! leak-audited inputs, probe excluded from grading, stratified lengths
//! enforced by construction, Wilson 95% CIs, paired McNemar tests,
//! per-bucket breakdown, multi-seed stability, and a budget sweep so the
//! canonical 350-token budget can't be a cherry-pick.
//!
//! Rerun it yourself: `cargo run --example bench`

use ctx::{estimate_tokens, Chronos, Compactor, Lethe, Message, Mnemosyne, Moirai, Role, Themis};
use std::time::Instant;

// ---------------------------------------------------------------------------
// Deterministic RNG (xorshift64*): reproducible eval set, no dependencies.
// ---------------------------------------------------------------------------

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

// ---------------------------------------------------------------------------
// Eval set.
// ---------------------------------------------------------------------------

/// (early fact, answer keyword, L2 essentials rubric).
///
/// The keyword is the strict substring probe (L1). The essentials are the
/// human grading rubric for paraphrase-tolerant matching (L2): folded,
/// token-exact with plural tolerance (see `grade_l2_answer`). Essentials are
/// deliberately minimal — the words an answer must contain to count as
/// knowing the fact — and they are published here so the rubric is reviewable.
/// L2 tolerates rephrasing but NOT number-word translation ("500" must appear
/// as digits); that documented gap is what a future LLM-judge would fill.
const FACTS: &[(&str, &str, &[&str])] = &[
    (
        "Please remember this for later: my budget is under $500 for the whole trip.",
        "$500",
        &["500", "budget"],
    ),
    (
        "Keep in mind: I am allergic to peanuts, so avoid them in every suggestion.",
        "peanuts",
        &["peanut"],
    ),
    (
        "An important detail: my name is Amara and my deadline is Friday.",
        "Amara",
        &["amara", "friday"],
    ),
    (
        "Note this rule for the trip: I must not fly, I only travel by train.",
        "train",
        &["train"],
    ),
    (
        "Remember: I always carry an epi pen, and I prefer morning starts.",
        "epi pen",
        &["epi", "pen"],
    ),
];

const USER_FILLER: &[&str] = &[
    "that sounds fun, what else is around there",
    "got it, tell me more about the lake trail",
    "how long does the walk take at an easy pace",
    "is the path busy on weekends, do you know",
    "what should I pack for a day out there",
    "are there places to sit and rest along the way",
    "sounds nice, is it good for beginners too",
    "thanks, that helps a lot",
    "hey, what was that spot you mentioned before",
    "cool, and is there food nearby afterwards",
];

const ASSISTANT_FILLER: &[&str] = &[
    "hey there! happy to help with trip ideas",
    "the lake loop is gentle and takes about two hours",
    "there are benches near the water and shade in the afternoon",
    "mornings are quieter and the light is lovely",
    "bring water and a hat, the far stretch has little shade",
    "the cafe by the dock serves soup and sandwiches",
    "you can rent bikes near the south gate in summer",
    "the herons nest near the reeds in spring",
    "parking fills up fast so an early start helps",
    "the visitor cabin has maps and friendly staff",
];

const FINAL_QUESTIONS: &[&str] = &[
    "so, with all that in mind, what was the key thing I asked you to remember",
    "given everything I told you, what is the one detail I need you to hold on to",
    "what is the main point from the start that should shape your answer",
];

/// Long early message with the constraint embedded inside a wall of journal
/// prose: high total value, low value-per-token.
const JOURNAL_TEMPLATE: &str = "Here is the full journal from the last trip for context. \
Day one began with a late arrival, then a slow walk by the harbor and soup at the cafe by the dock. \
Day two was the lake loop at an easy pace, with long rests on the benches near the water. \
Day three brought rain, so the hours went to maps, books, and tea in the visitor cabin. \
Day four was bikes near the south gate, then herons near the reeds at dusk. \
The big lesson from those days concerns the budget: keep the whole trip under $500 \
or the rest falls apart. That ceiling shapes every choice below.";

/// L2 rubric for the journal needle (same keyword as FACTS[0] by design —
/// conversations are graded independently, so sharing is harmless).
const JOURNAL_ESSENTIALS: &[&str] = &["500", "budget"];

struct Conversation {
    id: String,
    history: Vec<Message>,
    keyword: String,
    essentials: &'static [&'static str],
    kind: &'static str,
}

fn build_conversations(seed: u64) -> Vec<Conversation> {
    let mut rng = Rng(seed);
    let mut convos = Vec::new();

    for i in 0..50 {
        let (kind, base_fillers, fact_text, keyword, essentials) = if i < 31 {
            // Short: everything fits. Both strategies should ace these.
            let (fact, keyword, essentials) = FACTS[i % FACTS.len()];
            (
                "short",
                4 + rng.below(3),
                fact.to_string(),
                keyword.to_string(),
                essentials,
            )
        } else if i < 47 {
            // Long: budget forces cuts. The early fact is the highest-merit
            // message, so merit retention keeps it while recency drops it.
            let (fact, keyword, essentials) = FACTS[i % FACTS.len()];
            (
                "long",
                26 + rng.below(7),
                fact.to_string(),
                keyword.to_string(),
                essentials,
            )
        } else {
            // Adversarial: the fact arrives as one long low-density journal.
            // Too big to pack whole, too old to survive the window — but its
            // best sentence can survive compression (see the bench results).
            (
                "adversarial",
                22,
                JOURNAL_TEMPLATE.to_string(),
                "$500".to_string(),
                JOURNAL_ESSENTIALS,
            )
        };

        let mut history = Vec::new();
        history.push(Message::user("msg_001", fact_text));
        for t in 0..base_fillers {
            let (role, text) = sample_filler(&mut rng, t.is_multiple_of(2));
            history.push(Message::new("", role, text));
        }
        history.push(Message::user(
            "",
            FINAL_QUESTIONS[rng.below(FINAL_QUESTIONS.len())],
        ));

        // Enforce the length regime by construction: shorts must fit the
        // canonical budget, longs/adversarials must exceed it with margin.
        // This is disclosed stratification (see benchmarks/README.md), not
        // tuning — no strategy's behavior influences bucket membership, and
        // for the canonical seed the loops below are no-ops.
        if kind == "short" {
            while history_tokens(&history) > CANONICAL_BUDGET && history.len() > 3 {
                history.remove(history.len() - 2); // drop filler, keep Q last
            }
        } else {
            let mut extra = 0usize;
            while history_tokens(&history) < CANONICAL_BUDGET + STRAT_MARGIN {
                let (role, text) =
                    sample_filler(&mut rng, (base_fillers + extra).is_multiple_of(2));
                extra += 1;
                history.insert(history.len() - 1, Message::new("", role, text));
            }
        }
        for (k, msg) in history.iter_mut().enumerate() {
            msg.id = format!("msg_{:03}", k + 1);
        }

        convos.push(Conversation {
            id: format!("convo_{:02}", i + 1),
            history,
            keyword,
            essentials,
            kind,
        });
    }
    convos
}

/// Canonical budget the length strata are defined against. Sweeps measure
/// other budgets, but bucket membership never changes.
const CANONICAL_BUDGET: usize = 350;
/// Long/adversarial conversations must clear the canonical budget by at
/// least this much, so "forces cuts" holds with margin on every seed.
const STRAT_MARGIN: usize = 50;

fn sample_filler(rng: &mut Rng, assistant: bool) -> (Role, &'static str) {
    if assistant {
        (
            Role::Assistant,
            ASSISTANT_FILLER[rng.below(ASSISTANT_FILLER.len())],
        )
    } else {
        (Role::User, USER_FILLER[rng.below(USER_FILLER.len())])
    }
}

fn history_tokens(history: &[Message]) -> usize {
    history.iter().map(Message::tokens).sum()
}

// ---------------------------------------------------------------------------
// Runner: leak audit, grading, and statistics.
// ---------------------------------------------------------------------------

/// Fail loudly on eval-set leaks: the keyword may appear in the needle turn
/// (index 0) and in generated summaries, but never anywhere else in the
/// *input* — a filler or final question containing it would silently inflate
/// every strategy's score. Also rejects vacuous needles (keyword absent even
/// from turn 1), against which no strategy could score.
fn audit_no_leaks(convos: &[Conversation]) {
    for convo in convos {
        let needle = convo.keyword.to_lowercase();
        let mut needle_hits = 0usize;
        for (k, msg) in convo.history.iter().enumerate() {
            if msg.content.to_lowercase().contains(&needle) {
                if k == 0 {
                    needle_hits += 1;
                } else {
                    panic!(
                        "eval leak in {}: keyword {:?} appears in {} (not the needle turn)",
                        convo.id, convo.keyword, msg.id
                    );
                }
            }
        }
        assert!(
            needle_hits > 0,
            "vacuous needle in {}: keyword {:?} absent from turn 1",
            convo.id,
            convo.keyword
        );
    }
}

/// Grade one conversation: compact, then check the keyword survives anywhere
/// except the final turn. The final turn is the probe ("what did I tell
/// you?"), not evidence — excluding it is defense in depth for the day a
/// probe paraphrases the fact. (No current probe contains a keyword, so this
/// changes nothing today; the leak audit above enforces that.)
fn grade(strategy: &dyn Compactor, convo: &Conversation) -> (bool, usize) {
    let (kept, receipt) = strategy
        .compact(&convo.history)
        .expect("compaction is total");
    let probe_id = convo.history.last().map(|m| m.id.as_str()).unwrap_or("");
    let ok = kept.iter().filter(|m| m.id != probe_id).any(|m| {
        m.content
            .to_lowercase()
            .contains(&convo.keyword.to_lowercase())
    });
    (ok, receipt.output_tokens)
}

/// Wilson score interval (95%) for a binomial proportion. Point estimates
/// from n=50 are wide; publish the interval, not just the point.
pub fn wilson(p_hat: f64, n: usize) -> (f64, f64) {
    const Z: f64 = 1.96;
    let n = n as f64;
    let denom = 1.0 + Z * Z / n;
    let center = (p_hat + Z * Z / (2.0 * n)) / denom;
    let half = Z * (p_hat * (1.0 - p_hat) / n + Z * Z / (4.0 * n * n)).sqrt() / denom;
    ((center - half).max(0.0), (center + half).min(1.0))
}

/// Standard normal CDF via the Abramowitz–Stegun erf approximation.
pub fn erf(x: f64) -> f64 {
    if x == 0.0 {
        return 0.0;
    }
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    const A1: f64 = 0.254829592;
    const A2: f64 = -0.284496736;
    const A3: f64 = 1.421413741;
    const A4: f64 = -1.453152027;
    const A5: f64 = 1.061405429;
    const P: f64 = 0.3275911;
    let t = 1.0 / (1.0 + P * x);
    let poly = ((((A5 * t + A4) * t + A3) * t + A2) * t + A1) * t;
    sign * (1.0 - poly * (-x * x).exp())
}

pub fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

/// McNemar's test (continuity-corrected) for paired binary outcomes on the
/// same conversations: b = A-wrong/B-right, c = A-right/B-wrong.
/// Returns (chi-square, p-value). b + c == 0 means total agreement.
pub fn mcnemar(b: usize, c: usize) -> (f64, f64) {
    if b + c == 0 {
        return (0.0, 1.0);
    }
    let chi2 = ((b as f64 - c as f64).abs() - 1.0).powi(2) / (b + c) as f64;
    let p = 2.0 * (1.0 - normal_cdf(chi2.sqrt()));
    (chi2, p.clamp(0.0, 1.0))
}

fn make_strategies(budget: Chronos) -> Vec<(&'static str, Box<dyn Compactor>)> {
    vec![
        ("lethe", Box::new(Lethe::new(budget))),
        ("moirai", Box::new(Moirai::new(budget))),
        (
            "mnemosyne",
            Box::new(Mnemosyne::new(budget).with_recent_turns(6)),
        ),
    ]
}

/// Median per-compaction latency in ms: one warmup pass, then median of 10
/// timed passes. Debug-build, same-machine numbers — rerun with `--release`
/// for quotable latencies.
fn measure_latency(strategy: &dyn Compactor, convos: &[Conversation]) -> f64 {
    for convo in convos {
        let _ = strategy.compact(&convo.history);
    }
    let mut samples = Vec::new();
    for _ in 0..10 {
        let start = Instant::now();
        for convo in convos {
            let _ = strategy.compact(&convo.history);
        }
        samples.push(start.elapsed().as_secs_f64() * 1000.0 / convos.len() as f64);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    samples[samples.len() / 2]
}

/// Format a p-value for the table. Below 1e-4, report the bound instead —
/// a bare "p=0.0000" reads as impossibility, which a finite eval can never show.
fn fmt_p(p: f64) -> String {
    if p < 0.0001 {
        "p<0.0001".to_string()
    } else {
        format!("p={p:.4}")
    }
}

/// Discordant paired outcomes between two strategies' grade vectors:
/// (A-wrong/B-right, A-right/B-wrong).
fn discordant(a: &[bool], b: &[bool]) -> (usize, usize) {
    let mut bc = (0usize, 0usize);
    for (x, y) in a.iter().zip(b.iter()) {
        match (x, y) {
            (false, true) => bc.0 += 1,
            (true, false) => bc.1 += 1,
            _ => {}
        }
    }
    bc
}

// ---------------------------------------------------------------------------
// Real-model mode (`--with-model`): grade answers, not just retention.
// ---------------------------------------------------------------------------
//
// The text-retention eval above proves CTX keeps the right TEXT. This mode
// closes the remaining gap: it sends each compacted history to a real model
// backend for the final turn and grades the ANSWER. Additive to the no-model
// eval, never a replacement — run it with:
//
//   cargo run --example bench -- --with-model
//
// Environment (no key is ever printed, logged, or persisted):
//   CTX_MODEL_API_KEY (or GROQ_API_KEY, or OPENAI_API_KEY)   required
//   CTX_MODEL_BASE_URL  default https://api.groq.com/openai/v1
//   CTX_MODEL_NAME      default llama-3.3-70b-versatile
//   CTX_PRICE_IN_PER_M  default 0.59 (USD per 1M prompt tokens)
//   CTX_PRICE_OUT_PER_M default 0.79 (USD per 1M completion tokens)
//   CTX_MODEL_PAUSE_MS  default 1000 (spacing between calls)

/// Parsed CLI for model mode.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelOpts {
    pub enabled: bool,
    /// Prefix of the eval set to run through the model (default: all).
    pub limit: usize,
    /// Milliseconds to wait between model calls (rate-limit courtesy).
    pub pause_ms: u64,
}

pub fn parse_args_from(args: &[String]) -> Result<ModelOpts, String> {
    let mut opts = ModelOpts {
        enabled: false,
        limit: usize::MAX,
        pause_ms: 1000,
    };
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--with-model" => opts.enabled = true,
            "--model-limit" => {
                i += 1;
                opts.limit = args
                    .get(i)
                    .and_then(|s| s.parse::<usize>().ok())
                    .filter(|n| *n > 0)
                    .ok_or_else(|| "--model-limit needs a positive integer".to_string())?;
            }
            "--model-pause-ms" => {
                i += 1;
                opts.pause_ms = args
                    .get(i)
                    .and_then(|s| s.parse::<u64>().ok())
                    .ok_or_else(|| "--model-pause-ms needs an integer".to_string())?;
            }
            "--help" | "-h" => return Err("help".to_string()),
            other => return Err(format!("unknown argument: {other}")),
        }
        i += 1;
    }
    if (opts.limit != usize::MAX || opts.pause_ms != 1000) && !opts.enabled {
        return Err("--model-limit/--model-pause-ms require --with-model".to_string());
    }
    Ok(opts)
}

fn usage() -> &'static str {
    "usage: cargo run --example bench [-- --with-model [--model-limit N] [--model-pause-ms MS]]"
}

/// A model backend: takes the compacted history (INCLUDING the final user
/// turn — unlike retention grading, the model must see the question) and
/// returns the final answer. Exists so the eval never depends on one vendor:
/// implement this trait for any API.
pub trait ModelBackend {
    /// Human label for reports, e.g. "openai-compat groq/llama-3.3-70b-versatile".
    /// Must never contain key material.
    fn describe(&self) -> String;
    fn complete(&self, system: &str, history: &[Message]) -> Result<Completion, String>;
}

/// A finished model call.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Completion {
    pub text: String,
    pub prompt_tokens: Option<u64>,
    pub completion_tokens: Option<u64>,
    /// True when token counts fell back to the local estimator (the API
    /// reported no usage block). Costs built on estimates are marked as such.
    pub usage_estimated: bool,
}

/// The exact system prompt sent with every model call. Printed in the report
/// and persisted with the answers so the run is reproducible.
pub const MODEL_SYSTEM_PROMPT: &str = "Answer the user's final question using only the conversation history above. Reply in one or two sentences.";
pub const MODEL_TEMPERATURE: f64 = 0.0;
pub const MODEL_MAX_TOKENS: u32 = 256;

/// OpenAI-compatible chat-completions backend (Groq, OpenAI, and anything
/// speaking the same protocol), reached over HTTPS via a `curl` subprocess —
/// no HTTP client dependency. Bodies travel through temp files, never through
/// a shell, so quoting cannot corrupt them.
pub struct OpenAiCompatBackend {
    base_url: String,
    api_key: String,
    model: String,
    timeout_secs: u64,
}

impl OpenAiCompatBackend {
    /// Read configuration from the environment. The key is looked up as
    /// CTX_MODEL_API_KEY, then GROQ_API_KEY, then OPENAI_API_KEY, and is
    /// stored for the Authorization header only — never printed or persisted.
    pub fn from_env() -> Result<Self, String> {
        let api_key = std::env::var("CTX_MODEL_API_KEY")
            .or_else(|_| std::env::var("GROQ_API_KEY"))
            .or_else(|_| std::env::var("OPENAI_API_KEY"))
            .map_err(|_| {
                "no model API key is set (looked for CTX_MODEL_API_KEY, GROQ_API_KEY, OPENAI_API_KEY)"
                    .to_string()
            })?;
        if api_key.trim().is_empty() {
            return Err("model API key env var is set but empty".to_string());
        }
        let base_url = std::env::var("CTX_MODEL_BASE_URL")
            .unwrap_or_else(|_| "https://api.groq.com/openai/v1".to_string());
        let model = std::env::var("CTX_MODEL_NAME")
            .unwrap_or_else(|_| "llama-3.3-70b-versatile".to_string());
        Ok(Self {
            base_url,
            api_key,
            model,
            timeout_secs: 90,
        })
    }

    /// Override the per-call timeout (seconds). Mainly useful for tests
    /// against unreachable hosts.
    #[allow(dead_code)]
    pub fn with_timeout_secs(mut self, secs: u64) -> Self {
        self.timeout_secs = secs;
        self
    }

    fn role(role: Role) -> &'static str {
        match role {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Tool => "tool",
        }
    }
}

// Manual Debug: the key must never appear in logs or panic output.
impl std::fmt::Debug for OpenAiCompatBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAiCompatBackend")
            .field("base_url", &self.base_url)
            .field("model", &self.model)
            .field("api_key", &"***")
            .finish()
    }
}

/// Escape a string for embedding in JSON (quotes, backslashes, controls).
/// Non-ASCII passes through raw: the body is written as UTF-8, which JSON
/// accepts.
pub fn json_escape(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 2);
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0C}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

/// Decode one JSON string literal starting just after the opening quote.
/// Returns (decoded, bytes consumed including the closing quote).
fn decode_json_string(bytes: &[u8]) -> Option<(String, usize)> {
    let mut out = String::new();
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'"' => return Some((out, i + 1)),
            b'\\' => {
                i += 1;
                match bytes.get(i) {
                    Some(b'"') => out.push('"'),
                    Some(b'\\') => out.push('\\'),
                    Some(b'/') => out.push('/'),
                    Some(b'b') => out.push('\u{08}'),
                    Some(b'f') => out.push('\u{0C}'),
                    Some(b'n') => out.push('\n'),
                    Some(b'r') => out.push('\r'),
                    Some(b't') => out.push('\t'),
                    Some(b'u') => {
                        if i + 4 >= bytes.len() {
                            return None;
                        }
                        let hex = std::str::from_utf8(&bytes[i + 1..i + 5]).ok()?;
                        let mut unit = u32::from_str_radix(hex, 16).ok()?;
                        i += 4;
                        // Combine surrogate pairs.
                        if (0xD800..0xDC00).contains(&unit)
                            && bytes.get(i + 1) == Some(&b'\\')
                            && bytes.get(i + 2) == Some(&b'u')
                            && i + 6 < bytes.len()
                        {
                            if let Some(low) = std::str::from_utf8(&bytes[i + 3..i + 7])
                                .ok()
                                .and_then(|h| u32::from_str_radix(h, 16).ok())
                                .filter(|l| (0xDC00..0xE000).contains(l))
                            {
                                unit = 0x10000 + ((unit - 0xD800) << 10) + (low - 0xDC00);
                                i += 6;
                            }
                        }
                        out.push(char::from_u32(unit).unwrap_or('\u{FFFD}'));
                    }
                    _ => return None,
                }
            }
            _ => {
                // Regular UTF-8 char: take the whole code point.
                let s = std::str::from_utf8(&bytes[i..]).ok()?;
                let c = s.chars().next()?;
                out.push(c);
                i += c.len_utf8() - 1;
            }
        }
        i += 1;
    }
    None
}

/// Find `"key"` at any depth and return its decoded string value. Depth-first,
/// first match wins — sufficient for the fixed response shapes used here, and
/// covered by fixture tests below (see tests/model_eval.rs).
pub fn find_json_string(body: &str, key: &str) -> Option<String> {
    let pattern = format!("\"{key}\"");
    let bytes = body.as_bytes();
    let mut from = 0;
    while let Some(pos) = body[from..].find(&pattern) {
        let mut i = from + pos + pattern.len();
        while bytes.get(i).is_some_and(|b| b.is_ascii_whitespace()) {
            i += 1;
        }
        if bytes.get(i) == Some(&b':') {
            i += 1;
            while bytes.get(i).is_some_and(|b| b.is_ascii_whitespace()) {
                i += 1;
            }
            if bytes.get(i) == Some(&b'"') {
                if let Some((value, _)) = decode_json_string(&bytes[i + 1..]) {
                    return Some(value);
                }
            }
        }
        from = from + pos + 1;
    }
    None
}

/// Find `"key"` and parse its unsigned integer value (allows whitespace).
pub fn find_json_u64(body: &str, key: &str) -> Option<u64> {
    let pattern = format!("\"{key}\"");
    let bytes = body.as_bytes();
    let mut from = 0;
    while let Some(pos) = body[from..].find(&pattern) {
        let mut i = from + pos + pattern.len();
        while bytes.get(i).is_some_and(|b| b.is_ascii_whitespace()) {
            i += 1;
        }
        if bytes.get(i) == Some(&b':') {
            i += 1;
            while bytes.get(i).is_some_and(|b| b.is_ascii_whitespace()) {
                i += 1;
            }
            let start = i;
            while bytes.get(i).is_some_and(|b| b.is_ascii_digit()) {
                i += 1;
            }
            if i > start {
                if let Ok(n) = body[start..i].parse::<u64>() {
                    return Some(n);
                }
            }
        }
        from = from + pos + 1;
    }
    None
}

impl ModelBackend for OpenAiCompatBackend {
    fn describe(&self) -> String {
        format!(
            "openai-compat base={} model={} temp={} max_tokens={}",
            self.base_url, self.model, MODEL_TEMPERATURE, MODEL_MAX_TOKENS
        )
    }

    fn complete(&self, system: &str, history: &[Message]) -> Result<Completion, String> {
        let mut messages = format!(
            "{{\"role\":\"system\",\"content\":\"{}\"}}",
            json_escape(system)
        );
        for msg in history {
            messages.push_str(&format!(
                ",{{\"role\":\"{}\",\"content\":\"{}\"}}",
                Self::role(msg.role),
                json_escape(&msg.content)
            ));
        }
        let body = format!(
            "{{\"model\":\"{}\",\"messages\":[{}],\"temperature\":{},\"max_tokens\":{}}}",
            json_escape(&self.model),
            messages,
            MODEL_TEMPERATURE,
            MODEL_MAX_TOKENS
        );

        // Via temp file: argv-exact, no shell, no quoting hazards.
        let pid = std::process::id();
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.subsec_nanos())
            .unwrap_or(0);
        let path = std::env::temp_dir().join(format!("ctx_model_body_{pid}_{nonce}.json"));
        std::fs::write(&path, body.as_bytes())
            .map_err(|e| format!("cannot write request body: {e}"))?;
        let output = std::process::Command::new("curl.exe")
            .arg("-s")
            .arg("--max-time")
            .arg(self.timeout_secs.to_string())
            .arg("-X")
            .arg("POST")
            .arg(format!("{}/chat/completions", self.base_url))
            .arg("-H")
            .arg(format!("Authorization: Bearer {}", self.api_key))
            .arg("-H")
            .arg("Content-Type: application/json")
            .arg("--data-binary")
            .arg(format!("@{}", path.display()))
            .arg("-w")
            .arg("\nCTX_STATUS:%{http_code}")
            .output();
        let _ = std::fs::remove_file(&path);
        let output = output.map_err(|e| format!("cannot run curl.exe: {e}"))?;
        let stdout = String::from_utf8_lossy(&output.stdout);
        let (payload, status) = match stdout.rsplit_once("CTX_STATUS:") {
            Some((p, s)) => (p, s.trim().to_string()),
            None => (&stdout as &str, String::new()),
        };
        if status != "200" {
            let detail: String = payload.chars().take(300).collect();
            return Err(format!("backend HTTP {status}: {detail}"));
        }
        if let Some(message) = find_json_string(payload, "error") {
            return Err(format!("backend error envelope: {message}"));
        }
        // Some providers nest the message under choices[0]; a top-level
        // "content" would be a false hit, so require the choices marker.
        let text = if payload.contains("\"choices\"") {
            find_json_string(payload, "content")
        } else {
            None
        };
        let text = text.ok_or_else(|| {
            let detail: String = payload.chars().take(300).collect();
            format!("no content in backend response: {detail}")
        })?;
        let prompt_tokens = find_json_u64(payload, "prompt_tokens");
        let completion_tokens = find_json_u64(payload, "completion_tokens");
        let usage_estimated = prompt_tokens.is_none() || completion_tokens.is_none();
        let prompt_tokens = Some(prompt_tokens.unwrap_or_else(|| {
            estimate_tokens(system) as u64
                + history
                    .iter()
                    .map(|m| m.content.len() as u64 / 4 + 5)
                    .sum::<u64>()
        }));
        let completion_tokens =
            Some(completion_tokens.unwrap_or_else(|| (text.len() / 4) as u64 + 1));
        Ok(Completion {
            text,
            prompt_tokens,
            completion_tokens,
            usage_estimated,
        })
    }
}

// ---------------------------------------------------------------------------
// Answer graders: L1 (strict) and L2 (paraphrase-tolerant).
// ---------------------------------------------------------------------------

/// Fold one token for comparison: lowercase, alphanumeric only.
/// "$500" -> "500", "trains!" stays "trains".
pub fn fold_token(token: &str) -> String {
    token
        .to_lowercase()
        .chars()
        .filter(|c| c.is_alphanumeric())
        .collect()
}

/// Split an answer into folded alphanumeric runs. Crucially, hyphens and
/// other punctuation SPLIT rather than vanish: "epi-pen" yields "epi" +
/// "pen" (two rubric essentials), never the merged "epipen" that would
/// match neither.
pub fn answer_tokens(answer: &str) -> Vec<String> {
    answer
        .to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect()
}

/// An essential matches a folded answer token exactly, or with a plural
/// suffix — so "peanuts" satisfies "peanut", while "open"/"happen" never
/// satisfy "pen", and "5000" never satisfies "500".
pub fn token_matches(folded_token: &str, essential: &str) -> bool {
    folded_token == essential
        || *folded_token == format!("{essential}s")
        || *folded_token == format!("{essential}es")
}

/// L1: legacy strict probe — the raw keyword as a case-insensitive substring.
/// Cheap and continuous with the retention eval, but blind to paraphrase and
/// fooled by longer numbers ("5000" contains "500").
pub fn grade_l1_answer(answer: &str, keyword: &str) -> bool {
    answer.to_lowercase().contains(&keyword.to_lowercase())
}

/// L2: every rubric essential must token-match. Tolerates rephrasing, word
/// order, extra words, and plurals — but numerals must appear AS digits
/// ("five hundred" fails "500"): the documented gap an LLM-judge would fill.
pub fn grade_l2_answer(answer: &str, essentials: &[&str]) -> bool {
    let tokens = answer_tokens(answer);
    essentials
        .iter()
        .all(|e| tokens.iter().any(|t| token_matches(t, e)))
}

/// Retention-vs-answer outcome for one conversation under one strategy.
/// Both off-diagonal cells are findings, not noise (see the report).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Divergence {
    /// Text kept, answer right — the pipeline worked end to end.
    BothRight,
    /// Text kept, answer wrong — compaction succeeded, generation failed.
    RetainedButWrong,
    /// Text dropped, answer right — inference from context, or luck.
    DroppedButRight,
    /// Text dropped, answer wrong — the expected failure.
    BothWrong,
}

pub fn classify(retained: bool, answered: bool) -> Divergence {
    match (retained, answered) {
        (true, true) => Divergence::BothRight,
        (true, false) => Divergence::RetainedButWrong,
        (false, true) => Divergence::DroppedButRight,
        (false, false) => Divergence::BothWrong,
    }
}

/// USD cost of one call from token usage and per-1M-token prices.
pub fn call_cost(
    prompt_tokens: u64,
    completion_tokens: u64,
    price_in_per_m: f64,
    price_out_per_m: f64,
) -> f64 {
    prompt_tokens as f64 / 1_000_000.0 * price_in_per_m
        + completion_tokens as f64 / 1_000_000.0 * price_out_per_m
}

/// One evaluated model call. Errors are data: a failed call records `error`
/// and no grades, counts loudly in the report, and fails the run at the end
/// (exit 1) so partial results can never be mistaken for complete ones.
struct ModelRow {
    pub convo_id: String,
    pub kind: &'static str,
    pub strategy: &'static str,
    pub retained: bool,
    pub answer: Option<String>,
    pub l1: Option<bool>,
    pub l2: Option<bool>,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub usage_estimated: bool,
    pub latency_ms: f64,
    pub error: Option<String>,
}

/// Run the model eval: for each conversation (up to `limit`) and each
/// strategy, compact, send the FULL kept history (probe included — the model
/// must see the question), and grade the answer both ways.
#[allow(clippy::too_many_arguments)]
fn run_model_eval(
    backend: &dyn ModelBackend,
    convos: &[Conversation],
    strategies: &[(&'static str, Box<dyn Compactor>)],
    limit: usize,
    pause_ms: u64,
    price_in_per_m: f64,
    price_out_per_m: f64,
) -> (Vec<ModelRow>, f64) {
    let mut rows = Vec::new();
    let mut total_cost = 0.0;
    let n = convos.len().min(limit);
    for convo in convos.iter().take(n) {
        for (name, strategy) in strategies {
            let start = std::time::Instant::now();
            let (kept, _) = strategy
                .compact(&convo.history)
                .expect("compaction is total");
            let (retained, _) = grade(strategy.as_ref(), convo);
            let mut row = ModelRow {
                convo_id: convo.id.clone(),
                kind: convo.kind,
                strategy: name,
                retained,
                answer: None,
                l1: None,
                l2: None,
                prompt_tokens: 0,
                completion_tokens: 0,
                usage_estimated: false,
                latency_ms: 0.0,
                error: None,
            };
            match backend.complete(MODEL_SYSTEM_PROMPT, &kept) {
                Ok(done) => {
                    row.l1 = Some(grade_l1_answer(&done.text, &convo.keyword));
                    row.l2 = Some(grade_l2_answer(&done.text, convo.essentials));
                    row.prompt_tokens = done.prompt_tokens.unwrap_or(0);
                    row.completion_tokens = done.completion_tokens.unwrap_or(0);
                    row.usage_estimated = done.usage_estimated;
                    total_cost += call_cost(
                        row.prompt_tokens,
                        row.completion_tokens,
                        price_in_per_m,
                        price_out_per_m,
                    );
                    row.answer = Some(done.text);
                }
                Err(e) => {
                    row.error = Some(e);
                }
            }
            row.latency_ms = start.elapsed().as_secs_f64() * 1000.0;
            rows.push(row);
            if pause_ms > 0 {
                std::thread::sleep(std::time::Duration::from_millis(pause_ms));
            }
        }
    }
    (rows, total_cost)
}

/// Persist every answer with full run metadata for auditability. Answers are
/// synthetic-eval outputs (no PII, no key material — keys never enter this
/// file). JSON is written by hand with `json_escape`; corpus texts need no
/// further treatment.
fn save_model_answers(
    path: &str,
    backend_desc: &str,
    seed: u64,
    budget: usize,
    price_in_per_m: f64,
    price_out_per_m: f64,
    rows: &[ModelRow],
) -> Result<(), String> {
    let epoch = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let mut out = format!(
        "{{\"run\":{{\"backend\":\"{}\",\"seed\":{},\"budget\":{},\"system_prompt\":\"{}\",\"temperature\":{},\"max_tokens\":{},\"price_in_per_m\":{},\"price_out_per_m\":{},\"generated_epoch\":{}}},\"rows\":[",
        json_escape(backend_desc),
        seed,
        budget,
        json_escape(MODEL_SYSTEM_PROMPT),
        MODEL_TEMPERATURE,
        MODEL_MAX_TOKENS,
        price_in_per_m,
        price_out_per_m,
        epoch
    );
    for (k, row) in rows.iter().enumerate() {
        if k > 0 {
            out.push(',');
        }
        let answer = row.answer.as_deref().unwrap_or("");
        let error = row.error.as_deref().unwrap_or("");
        out.push_str(&format!(
            "{{\"convo\":\"{}\",\"kind\":\"{}\",\"strategy\":\"{}\",\"retained\":{},\"answer\":\"{}\",\"l1\":{},\"l2\":{},\"prompt_tokens\":{},\"completion_tokens\":{},\"usage_estimated\":{},\"latency_ms\":{:.1},\"error\":\"{}\"}}",
            json_escape(&row.convo_id),
            json_escape(row.kind),
            json_escape(row.strategy),
            row.retained,
            json_escape(answer),
            row.l1.map(|b| b.to_string()).unwrap_or_else(|| "null".to_string()),
            row.l2.map(|b| b.to_string()).unwrap_or_else(|| "null".to_string()),
            row.prompt_tokens,
            row.completion_tokens,
            row.usage_estimated,
            row.latency_ms,
            json_escape(error)
        ));
    }
    out.push_str("]}");
    std::fs::write(path, out).map_err(|e| format!("cannot write {path}: {e}"))?;
    Ok(())
}

fn main() {
    const CANONICAL_SEED: u64 = 1701;
    const SEEDS: &[u64] = &[1701, 7, 99, 1234, 9001];
    const SWEEP_BUDGETS: &[usize] = &[80, 150, 350, 600, 1000];

    let raw_args: Vec<String> = std::env::args().skip(1).collect();
    let model_opts = match parse_args_from(&raw_args) {
        Ok(opts) => opts,
        Err(e) => {
            if e == "help" {
                println!("{}", usage());
                println!("\nEnvironment:\n  CTX_MODEL_API_KEY (or GROQ_API_KEY, or OPENAI_API_KEY)\n  CTX_MODEL_BASE_URL  (default https://api.groq.com/openai/v1)\n  CTX_MODEL_NAME      (default llama-3.3-70b-versatile)\n  CTX_PRICE_IN_PER_M / CTX_PRICE_OUT_PER_M (default 0.59 / 0.79 USD)");
                std::process::exit(0);
            }
            eprintln!("error: {e}\n{}", usage());
            std::process::exit(2);
        }
    };

    // Fail fast on model misconfiguration: never burn through the text eval
    // only to discover the key was missing, and never silently continue
    // without the model when it was requested.
    let backend: Option<OpenAiCompatBackend> = if model_opts.enabled {
        match OpenAiCompatBackend::from_env() {
            Ok(b) => Some(b),
            Err(e) => {
                eprintln!(
                    "ERROR: --with-model was requested but the backend is unusable: {e}.\n\
                     Set CTX_MODEL_API_KEY (or GROQ_API_KEY / OPENAI_API_KEY) and, if needed,\n\
                     CTX_MODEL_BASE_URL + CTX_MODEL_NAME. Refusing to silently fall back\n\
                     to the no-model eval."
                );
                std::process::exit(2);
            }
        }
    } else {
        None
    };

    println!(
        "CTX benchmark — needle-retention eval (no model in the loop; see benchmarks/README.md)"
    );
    println!(
        "50 synthetic conversations x {} seeds; canonical seed {CANONICAL_SEED}, canonical budget {CANONICAL_BUDGET} tokens\n",
        SEEDS.len()
    );

    // ---- Headline: canonical seed, canonical budget. ----
    let canonical = build_conversations(CANONICAL_SEED);
    audit_no_leaks(&canonical);
    let strategies = make_strategies(Chronos::new(CANONICAL_BUDGET, 0));

    let mut grades: Vec<Vec<bool>> = Vec::new();
    println!(
        "{:<10} {:>7} {:>7} {:>9} {:>15} {:>11} {:>11}",
        "strategy", "correct", "total", "retained", "Wilson95%", "avgTok", "medMs"
    );
    for (name, strategy) in &strategies {
        let mut correct = 0usize;
        let mut total_out = 0usize;
        let mut g = Vec::new();
        for convo in &canonical {
            let (ok, tokens) = grade(strategy.as_ref(), convo);
            g.push(ok);
            if ok {
                correct += 1;
            }
            total_out += tokens;
        }
        grades.push(g);
        let n = canonical.len();
        let (lo, hi) = wilson(correct as f64 / n as f64, n);
        let ms = measure_latency(strategy.as_ref(), &canonical);
        println!(
            "{:<10} {:>7} {:>7} {:>8.1}% [{:>4.1}%, {:>4.1}%] {:>11.1} {:>11.3}",
            name,
            correct,
            n,
            100.0 * correct as f64 / n as f64,
            100.0 * lo,
            100.0 * hi,
            total_out as f64 / n as f64,
            ms
        );
    }

    // Paired significance: same conversations, so McNemar, not chi-square.
    let names: Vec<&str> = strategies.iter().map(|(name, _)| *name).collect();
    for (a, b) in [(0usize, 1usize), (1, 2), (0, 2)] {
        let (d1, d2) = discordant(&grades[a], &grades[b]);
        let (chi2, p) = mcnemar(d1, d2);
        println!(
            "McNemar {} vs {}: discordant {}/{} (wrong/right), chi2={:.2}, {}",
            names[a],
            names[b],
            d1,
            d2,
            chi2,
            fmt_p(p)
        );
    }

    // ---- Where each strategy wins and loses (canonical seed). ----
    println!("\nretention by bucket (canonical seed):");
    println!(
        "{:<12} {:>4} {:>8} {:>8} {:>10}",
        "bucket", "n", "lethe", "moirai", "mnemosyne"
    );
    for kind in ["short", "long", "adversarial"] {
        let idx: Vec<usize> = canonical
            .iter()
            .enumerate()
            .filter(|(_, c)| c.kind == kind)
            .map(|(k, _)| k)
            .collect();
        print!("{:<12} {:>4}", kind, idx.len());
        for g in &grades {
            let c = idx.iter().filter(|k| g[**k]).count();
            print!(" {:>7.1}%", 100.0 * c as f64 / idx.len() as f64);
        }
        println!();
    }

    // ---- Stability: same eval, more seeds (canonical budget). ----
    println!("\nstability across seeds (budget {CANONICAL_BUDGET}): mean [min, max] retained");
    for (si, (name, _)) in strategies.iter().enumerate() {
        let mut accs = Vec::new();
        for seed in SEEDS {
            let convos = build_conversations(*seed);
            audit_no_leaks(&convos);
            let strat = make_strategies(Chronos::new(CANONICAL_BUDGET, 0));
            let c = convos
                .iter()
                .filter(|cv| grade(strat[si].1.as_ref(), cv).0)
                .count();
            accs.push(100.0 * c as f64 / convos.len() as f64);
        }
        let mean = accs.iter().sum::<f64>() / accs.len() as f64;
        let min = accs.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = accs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        println!("{name:<10} {mean:>5.1}% [{min:>5.1}%, {max:>5.1}%]");
    }

    // ---- Sensitivity: same eval, more budgets (canonical seed). ----
    // The honest curve: the merit gap should live in the middle regime and
    // vanish where everything fits (top) or nothing fits (bottom).
    println!("\nbudget sweep (canonical seed): % retained per budget");
    print!("{:<8}", "budget");
    for (name, _) in &strategies {
        print!(" {:>8}", name);
    }
    println!();
    for b in SWEEP_BUDGETS {
        let strat = make_strategies(Chronos::new(*b, 0));
        print!("{:<8}", b);
        for (_, s) in &strat {
            let c = canonical
                .iter()
                .filter(|cv| grade(s.as_ref(), cv).0)
                .count();
            print!(" {:>7.1}%", 100.0 * c as f64 / canonical.len() as f64);
        }
        println!();
    }

    // ---- Validity: is this eval capable of discriminating at all? ----
    // Needles must score high (else nothing could retain them on merit) and
    // filler must score low (else merit has nothing to select). The filler
    // pool was screened for hint substrings when written, so this second
    // number is a separation check on a screened pool — NOT a precision
    // claim about wild text. See benchmarks/README.md.
    let scorer = Themis::new();
    let mut needles_hit = 0usize;
    let mut filler_hit = 0usize;
    let mut filler = 0usize;
    for convo in &canonical {
        let n = convo.history.len();
        if scorer.score(&convo.history[0]).value >= 0.50 {
            needles_hit += 1;
        }
        for msg in &convo.history[1..n - 1] {
            filler += 1;
            if scorer.score(msg).value >= 0.50 {
                filler_hit += 1;
            }
        }
    }
    println!(
        "\nvalidity: leak audit PASS (keyword unique to turn 1 in all 50 inputs); probe excluded from grading"
    );
    println!(
        "themis separation (screened pool): needles {}/{:.0} >= 0.50, filler {}/{} >= 0.50",
        needles_hit,
        canonical.len(),
        filler_hit,
        filler
    );

    // Per-conversation detail for inspection.
    println!("\nconvo        kind         tokens  lethe   moirai  mnemosyne");
    let detail = make_strategies(Chronos::new(CANONICAL_BUDGET, 0));
    for convo in &canonical {
        let total = history_tokens(&convo.history);
        let mark = |s: &dyn Compactor| {
            if grade(s, convo).0 {
                "correct"
            } else {
                "WRONG  "
            }
        };
        println!(
            "{:<12} {:<12} {:>7} {:>7} {:>7} {:>7}",
            convo.id,
            convo.kind,
            total,
            mark(detail[0].1.as_ref()),
            mark(detail[1].1.as_ref()),
            mark(detail[2].1.as_ref())
        );
    }

    // ---- Real-model answer eval (additive; default off). ----
    if let Some(backend) = backend {
        let code = run_model_section(
            &backend,
            &canonical,
            &strategies,
            CANONICAL_SEED,
            &model_opts,
        );
        std::process::exit(code);
    }
}

/// Median of a list (sorted copy; 0.0 when empty).
fn median_sorted(mut xs: Vec<f64>) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    xs[xs.len() / 2]
}

fn env_price(name: &str, default: f64) -> Result<f64, String> {
    match std::env::var(name) {
        Ok(v) => v
            .parse::<f64>()
            .map_err(|_| format!("{name}={v:?} is not a number")),
        Err(_) => Ok(default),
    }
}

/// The `--with-model` section. Returns the process exit code: 0 when every
/// call succeeded, 1 when any call failed (after printing the full report,
/// so failures are visible, never silent).
fn run_model_section(
    backend: &dyn ModelBackend,
    canonical: &[Conversation],
    strategies: &[(&'static str, Box<dyn Compactor>)],
    seed: u64,
    opts: &ModelOpts,
) -> i32 {
    let price_in = match env_price("CTX_PRICE_IN_PER_M", 0.59) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("error: {e}");
            return 2;
        }
    };
    let price_out = match env_price("CTX_PRICE_OUT_PER_M", 0.79) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("error: {e}");
            return 2;
        }
    };
    let n = canonical.len().min(opts.limit);
    println!(
        "\n==== Real-model answer eval (n={n}, {}) ====",
        backend.describe()
    );
    println!("system prompt: \"{MODEL_SYSTEM_PROMPT}\" (temperature {MODEL_TEMPERATURE}, max_tokens {MODEL_MAX_TOKENS})");
    println!("graders: L1 = keyword substring (strict baseline), L2 = rubric essentials (paraphrase-tolerant)");

    let (rows, total_cost) = run_model_eval(
        backend,
        canonical,
        strategies,
        opts.limit,
        opts.pause_ms,
        price_in,
        price_out,
    );
    let errors = rows.iter().filter(|r| r.error.is_some()).count();
    let estimated = rows
        .iter()
        .filter(|r| r.usage_estimated && r.error.is_none())
        .count();

    // Per-strategy answer accuracy (both graders) + cost + latency.
    println!(
        "\n{:<10} {:>7} {:>7} {:>7} {:>7} {:>9} {:>11} {:>9}",
        "strategy", "n", "L1", "L2", "agree", "cost/call", "medMs", "tokIn+Out"
    );
    let mut l2_vectors: Vec<Vec<bool>> = Vec::new();
    for (name, _) in strategies {
        let mine: Vec<&ModelRow> = rows.iter().filter(|r| r.strategy == *name).collect();
        let graded: Vec<&ModelRow> = mine.iter().filter(|r| r.error.is_none()).copied().collect();
        let l1 = graded.iter().filter(|r| r.l1 == Some(true)).count();
        let l2 = graded.iter().filter(|r| r.l2 == Some(true)).count();
        let agree = graded.iter().filter(|r| r.l1 == r.l2).count();
        let cost: f64 = graded
            .iter()
            .map(|r| call_cost(r.prompt_tokens, r.completion_tokens, price_in, price_out))
            .sum();
        let med = median_sorted(graded.iter().map(|r| r.latency_ms).collect());
        let toks: u64 = graded
            .iter()
            .map(|r| r.prompt_tokens + r.completion_tokens)
            .sum();
        let g = graded.len().max(1) as f64;
        println!(
            "{:<10} {:>7} {:>6.1}% {:>6.1}% {:>6.1}% {:>8.4}$ {:>11.1} {:>9}",
            name,
            graded.len(),
            100.0 * l1 as f64 / g,
            100.0 * l2 as f64 / g,
            100.0 * agree as f64 / g,
            cost / g,
            med,
            toks
        );
        l2_vectors.push(graded.iter().map(|r| r.l2 == Some(true)).collect());
    }
    // Paired significance on L2 answers (lethe vs moirai).
    if l2_vectors.len() >= 2 && l2_vectors[0].len() == l2_vectors[1].len() {
        let (d1, d2) = discordant(&l2_vectors[0], &l2_vectors[1]);
        let (chi2, p) = mcnemar(d1, d2);
        println!(
            "McNemar (L2 answers) lethe vs moirai: discordant {d1}/{d2}, chi2={chi2:.2}, {}",
            fmt_p(p)
        );
    }

    // Retention-vs-answer on the SAME conversations, per strategy.
    println!("\nretention (text) vs answer (L2) — same conversations:");
    println!(
        "{:<10} {:>10} {:>16} {:>16} {:>10}",
        "strategy", "bothRight", "retainedButWrong", "droppedButRight", "bothWrong"
    );
    for (name, _) in strategies {
        let mut cells = [0usize; 4];
        for row in rows
            .iter()
            .filter(|r| r.strategy == *name && r.error.is_none())
        {
            let answered = row.l2 == Some(true);
            cells[classify(row.retained, answered) as usize] += 1;
        }
        println!(
            "{:<10} {:>10} {:>16} {:>16} {:>10}",
            name, cells[0], cells[1], cells[2], cells[3]
        );
    }
    // The divergences, listed — these are the findings, not noise.
    for (name, _) in strategies {
        let rbw: Vec<&str> = rows
            .iter()
            .filter(|r| {
                r.strategy == *name
                    && r.error.is_none()
                    && classify(r.retained, r.l2 == Some(true)) == Divergence::RetainedButWrong
            })
            .map(|r| r.convo_id.as_str())
            .collect();
        let dbr: Vec<&str> = rows
            .iter()
            .filter(|r| {
                r.strategy == *name
                    && r.error.is_none()
                    && classify(r.retained, r.l2 == Some(true)) == Divergence::DroppedButRight
            })
            .map(|r| r.convo_id.as_str())
            .collect();
        if !rbw.is_empty() {
            println!(
                "retained-but-wrong ({name}; compaction ok, generation failed): {}",
                rbw.join(", ")
            );
        }
        if !dbr.is_empty() {
            println!(
                "dropped-but-right ({name}; inference or luck): {}",
                dbr.join(", ")
            );
        }
    }

    let path = "benchmarks/model_answers.json";
    match save_model_answers(
        path,
        &backend.describe(),
        seed,
        CANONICAL_BUDGET,
        price_in,
        price_out,
        &rows,
    ) {
        Ok(()) => println!("\nsaved {path} ({} rows, {} errors)", rows.len(), errors),
        Err(e) => {
            eprintln!("error: {e}");
            return 1;
        }
    }
    println!(
        "model spend total: ${total_cost:.4} ({} rows; usage {}reported by API{})",
        rows.len(),
        if estimated == 0 { "" } else { "PARTLY " },
        if estimated == 0 {
            ""
        } else {
            " — costs on estimated rows are approximate"
        }
    );
    if errors > 0 {
        eprintln!("ERROR: {errors} model call(s) failed — results incomplete, refusing exit 0.");
        return 1;
    }
    0
}
