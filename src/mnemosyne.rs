//! Mnemosyne — titan of memory. Nothing discarded, only made smaller.
//!
//! Old turns are folded into one short summary; recent turns stay verbatim on
//! top. Best quality-per-token when you can afford the summarization step.
//!
//! The summarizer is a trait: bring a model call in production
//! ([`Summarizer`]), use the shipped [`ExtractiveSummarizer`] when you want
//! zero network and fully deterministic output.

use crate::tokens::MESSAGE_OVERHEAD;
use crate::{Chronos, Clio, Compactor, Error, Message, Receipt, Result, Role, Themis};

/// The folded past: short text plus the load-bearing facts it preserves.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Summary {
    /// The compressed text that replaces the old turns.
    pub content: String,
    /// Key facts preserved, as short labels (e.g. `budget under $500`).
    /// Logged by Clio so the compression is auditable.
    pub key_facts: Vec<String>,
}

/// Folds old turns into a [`Summary`]. Implement this with a model call for
/// best quality; the default [`ExtractiveSummarizer`] needs no model.
///
/// `max_tokens` budgets the summary *content* only — the caller holds back
/// the summary message's framing overhead separately.
pub trait Summarizer {
    /// Compress `old` (chronological) into a summary of at most
    /// `max_tokens` (estimated).
    fn summarize(&self, old: &[Message], max_tokens: usize) -> Summary;
}

/// Deterministic, dependency-free summarizer: splits old turns into
/// sentences and keeps the highest merit-density sentences verbatim as
/// bullets, highest-merit first, until the token allowance runs out.
/// Sentence granularity is what lets one load-bearing sentence survive when
/// its whole turn is too diffuse to pack. No model call, no hallucinated
/// facts — every word in the summary was in the original.
#[derive(Clone, Copy, Debug)]
pub struct ExtractiveSummarizer {
    scorer: Themis,
}

impl ExtractiveSummarizer {
    /// Build it.
    #[must_use]
    pub fn new() -> Self {
        Self {
            scorer: Themis::new(),
        }
    }
}

impl Default for ExtractiveSummarizer {
    fn default() -> Self {
        Self::new()
    }
}

impl Summarizer for ExtractiveSummarizer {
    fn summarize(&self, old: &[Message], max_tokens: usize) -> Summary {
        if old.is_empty() || max_tokens == 0 {
            return Summary {
                content: String::new(),
                key_facts: Vec::new(),
            };
        }

        // Flatten old turns into (turn, sentence) pairs. A long turn that is
        // diffuse overall can still contribute its one load-bearing sentence.
        let mut sentences: Vec<(usize, String)> = Vec::new();
        for (i, msg) in old.iter().enumerate() {
            for sentence in split_sentences(msg.content.trim()) {
                sentences.push((i, sentence));
            }
        }
        if sentences.is_empty() {
            return Summary {
                content: String::new(),
                key_facts: Vec::new(),
            };
        }

        // Rank sentences by merit density, oldest first on ties. Each
        // sentence is scored with its turn's role: a user's phrasing binds
        // even in a single sentence.
        let mut order: Vec<usize> = (0..sentences.len()).collect();
        order.sort_by(|&a, &b| {
            let (ia, sa) = &sentences[a];
            let (ib, sb) = &sentences[b];
            let probe_a = Message::new("", old[*ia].role, sa.clone());
            let probe_b = Message::new("", old[*ib].role, sb.clone());
            let da = self
                .scorer
                .score(&probe_a)
                .density(crate::tokens::estimate_tokens(sa).max(1));
            let db = self
                .scorer
                .score(&probe_b)
                .density(crate::tokens::estimate_tokens(sb).max(1));
            db.partial_cmp(&da)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.cmp(&b))
        });

        let mut lines = Vec::new();
        let mut key_facts = Vec::new();
        // Content-only accounting: the summary message's framing overhead is
        // reserved by the caller (see compact_with), so it isn't counted here.
        let mut used = crate::tokens::estimate_tokens("Earlier summary:");
        for &i in &order {
            let (turn, sentence) = &sentences[i];
            let probe = Message::new("", old[*turn].role, sentence.clone());
            let importance = self.scorer.score(&probe);
            let bullet = format!("- [{}] {}", old[*turn].id, sentence);
            let cost = crate::tokens::estimate_tokens(&bullet) + 1;
            if used + cost > max_tokens {
                continue;
            }
            used += cost;
            // Key facts are the high-importance sentences, trimmed to a label.
            if importance.value >= 0.5 && key_facts.len() < 5 {
                key_facts.push(short_fact(sentence));
            }
            lines.push(bullet);
            if key_facts.len() >= 5 && used + 10 > max_tokens {
                break;
            }
        }

        if lines.is_empty() {
            // Allowance is tiny: preserve the single most important sentence,
            // truncated, rather than returning nothing. If truncation ate the
            // sentence, no key fact is claimed — Clio logs `[]`, an honest
            // signal that compression bottomed out.
            let (turn, sentence) = &sentences[order[0]];
            let mut text = sentence.clone();
            truncate_to_tokens(&mut text, max_tokens.saturating_sub(8));
            if text == *sentence {
                key_facts.push(short_fact(&text));
            }
            lines.push(format!("- [{}] {}", old[*turn].id, text));
        }

        Summary {
            content: format!("Earlier summary:\n{}", lines.join("\n")),
            key_facts,
        }
    }
}

/// Memory that compresses instead of cutting.
#[derive(Clone, Copy, Debug)]
pub struct Mnemosyne {
    budget: Chronos,
    /// Recent turns kept verbatim, newest-last.
    recent_turns: usize,
    /// Id stamped on the generated summary message.
    summary_id: &'static str,
}

impl Mnemosyne {
    /// Build the strategy around Chronos's limit. Keeps the last 6 turns
    /// verbatim by default.
    #[must_use]
    pub fn new(budget: Chronos) -> Self {
        Self {
            budget,
            recent_turns: 6,
            summary_id: "msg_summary",
        }
    }

    /// How many of the newest turns stay verbatim (default 6).
    #[must_use]
    pub fn with_recent_turns(mut self, n: usize) -> Self {
        self.recent_turns = n.max(1);
        self
    }

    /// Compact with the default extractive summarizer (no model call).
    pub fn compact(&self, history: &[Message]) -> Result<(Vec<Message>, Receipt)> {
        self.compact_with(history, &ExtractiveSummarizer::new())
    }

    /// Compact with your own summarizer — e.g. one model call that folds the
    /// old turns into a short paragraph.
    pub fn compact_with<S: Summarizer>(
        &self,
        history: &[Message],
        summarizer: &S,
    ) -> Result<(Vec<Message>, Receipt)> {
        let budget = self.budget.budget();
        if budget == 0 && !history.is_empty() {
            return Err(Error::ZeroBudget);
        }

        let input_tokens: usize = history.iter().map(Message::tokens).sum();
        let mut clio = Clio::new();

        // Fast path: everything fits — no reason to compress anything.
        if input_tokens <= budget {
            for msg in history {
                clio.kept(
                    msg.id.clone(),
                    "fits within budget, kept verbatim".to_string(),
                );
            }
            return Ok((
                history.to_vec(),
                clio.finish(input_tokens, input_tokens, budget),
            ));
        }

        // Split: recent turns stay verbatim, old turns get folded.
        let split = history.len().saturating_sub(self.recent_turns);
        let (old, recent) = history.split_at(split);
        let recent_tokens: usize = recent.iter().map(Message::tokens).sum();

        // Degenerate case: even the recent window overflows. Fall back to a
        // sliding window over the recent turns so we always return something
        // usable, and say so on the record.
        if recent_tokens > budget {
            let mut kept_tokens = 0usize;
            let mut keep_from = recent.len();
            for (i, msg) in recent.iter().enumerate().rev() {
                let cost = msg.tokens();
                if kept_tokens + cost <= budget || keep_from == recent.len() {
                    keep_from = i;
                    kept_tokens += cost;
                    if keep_from == 0 {
                        break;
                    }
                } else {
                    break;
                }
            }
            for msg in old {
                clio.dropped(
                    msg.id.clone(),
                    "folded away — recent window alone exceeds budget".to_string(),
                );
            }
            for msg in &recent[..keep_from] {
                clio.dropped(
                    msg.id.clone(),
                    format!(
                        "sliding window cut inside recent turns — budget exceeded by {} tokens",
                        recent_tokens.saturating_sub(budget)
                    ),
                );
            }
            let kept: Vec<Message> = recent[keep_from..].to_vec();
            for msg in &kept {
                clio.kept(msg.id.clone(), "recent turn, kept verbatim".to_string());
            }
            let output_tokens: usize = kept.iter().map(Message::tokens).sum();
            return Ok((kept, clio.finish(input_tokens, output_tokens, budget)));
        }

        // Normal case: summarize the old turns into the remaining allowance.
        // The summarizer budgets content only; the summary message's framing
        // overhead is held back up front so Chronos's limit always holds.
        let allowance = (budget - recent_tokens).saturating_sub(MESSAGE_OVERHEAD);
        let summary = if old.is_empty() {
            Summary {
                content: String::new(),
                key_facts: Vec::new(),
            }
        } else {
            summarizer.summarize(old, allowance)
        };

        let mut kept = Vec::new();
        if !summary.content.is_empty() {
            let summary_msg = Message::new(self.summary_id, Role::System, summary.content.clone());
            let summary_tokens = summary_msg.tokens();
            kept.push(summary_msg);
            let range = Clio::range_label(&old[0].id, &old[old.len() - 1].id);
            clio.summarized(
                range,
                format!(
                    "compressed to {summary_tokens} tokens, key facts preserved: [{}]",
                    summary.key_facts.join(", ")
                ),
            );
        } else if !old.is_empty() {
            for msg in old {
                clio.dropped(
                    msg.id.clone(),
                    "no summarizer allowance left under budget".to_string(),
                );
            }
        }
        for msg in recent {
            clio.kept(msg.id.clone(), "recent turn, kept verbatim".to_string());
            kept.push(msg.clone());
        }

        let output_tokens: usize = kept.iter().map(Message::tokens).sum();
        Ok((kept, clio.finish(input_tokens, output_tokens, budget)))
    }
}

impl Compactor for Mnemosyne {
    fn name(&self) -> &'static str {
        "mnemosyne"
    }

    fn compact(&self, history: &[Message]) -> Result<(Vec<Message>, Receipt)> {
        Mnemosyne::compact(self, history)
    }
}

/// Trim text in place to roughly `max_tokens` (heuristic estimator).
fn truncate_to_tokens(text: &mut String, max_tokens: usize) {
    if max_tokens == 0 {
        text.clear();
        return;
    }
    let max_chars = max_tokens * 4;
    if text.chars().count() > max_chars {
        let truncated: String = text.chars().take(max_chars).collect();
        *text = truncated + "…";
    }
}

/// Split text into sentences on `.`, `!`, `?`, and newlines. Deliberately
/// naive — abbreviations over-split into fragments, which the density ranking
/// tolerates (fragments score low and sink). The delimiter stays with its
/// sentence; empties are dropped.
fn split_sentences(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = String::new();
    for c in text.chars() {
        current.push(c);
        if matches!(c, '.' | '!' | '?' | '\n') {
            let sentence = current.trim().to_string();
            if !sentence.is_empty() {
                out.push(sentence);
            }
            current = String::new();
        }
    }
    let tail = current.trim().to_string();
    if !tail.is_empty() {
        out.push(tail);
    }
    out
}

/// A short label for a preserved fact: trimmed content, single line.
fn short_fact(content: &str) -> String {
    let flat: String = content.split_whitespace().collect::<Vec<_>>().join(" ");
    let chars: Vec<char> = flat.chars().collect();
    if chars.len() <= 48 {
        flat
    } else {
        chars[..48].iter().collect::<String>() + "…"
    }
}
