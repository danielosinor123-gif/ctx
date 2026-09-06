//! Themis — the weighing of what each message is worth.
//!
//! Themis scores every message on its content: constraints, decisions, and
//! named entities score high; small talk scores low. The scoring is a
//! deterministic heuristic — no model call, no network, reproducible
//! everywhere — and every score carries a human-readable reason so Clio can
//! log *why* a message lived or died.

use crate::{Message, Role};

/// Small-talk markers. Short messages built from these score at the floor.
const SMALL_TALK: &[&str] = &[
    "hi",
    "hello",
    "hey",
    "thanks",
    "thank you",
    "lol",
    "haha",
    "ok",
    "okay",
    "cool",
    "nice",
    "great",
    "bye",
    "good morning",
    "good night",
    "good afternoon",
    "how are you",
    "what's up",
    "whats up",
    "sounds good",
    "got it",
    "sure",
    "yep",
    "nope",
    "all right",
    "alright",
];

/// Phrases that mark a user constraint, fact about themselves, or instruction
/// the model must not forget. Checked heaviest, in roughly this order.
const CONSTRAINT_HINTS: &[&str] = &[
    "my name is",
    "i am allergic",
    "allergic to",
    "budget",
    "must not",
    "mustn't",
    "cannot",
    "can't",
    "never",
    "always",
    "don't",
    "do not",
    "remember",
    "prefer",
    "only",
    "constraint",
    "requirement",
    "important",
    "deadline",
    "asap",
    "rule",
    "instruction",
    "follow",
    "make sure",
    "keep in mind",
    "note that",
    "my",
    "$",
    "no more than",
    "at most",
    "exactly",
    "must",
    "rather than",
    "instead",
    "i want",
    "i'd like",
    "i would like",
];

/// Phrases that mark a decision the conversation reached.
const DECISION_HINTS: &[&str] = &[
    "decided",
    "decision",
    "agreed",
    "agreement",
    "going with",
    "let's go with",
    "lets go with",
    "final choice",
    "we chose",
    "we choose",
    "conclusion",
    "plan is",
    "settled",
    "we will",
    "replacement",
    "refund",
];

/// Phrases that mark a problem report — damage, defects, failures. They are
/// usually why the conversation exists at all; losing them loses the plot.
/// Listed as full word forms: matching is on word boundaries (see
/// [`contains_hint`]), so stems are spelled out per inflection.
const PROBLEM_HINTS: &[&str] = &[
    "damage",
    "damaged",
    "break",
    "breaks",
    "breaking",
    "broke",
    "broken",
    "crack",
    "cracked",
    "cracks",
    "cracking",
    "defect",
    "defects",
    "defective",
    "missing",
    "wrong item",
    "wrong order",
    "not working",
    "doesn't work",
    "does not work",
    "stopped working",
    "never arrived",
    "didn't arrive",
    "did not arrive",
    "charged twice",
    "overcharge",
    "overcharged",
];

/// Themis's verdict on one message: a score plus the reason for it.
#[derive(Clone, Debug, PartialEq)]
pub struct Importance {
    /// 0.0–1.0. Constraints and decisions land high, small talk lands low.
    pub value: f32,
    /// Human-readable evidence, e.g. `contains user constraint: 'budget under
    /// $500'` or `small talk`. Logged verbatim by Clio.
    pub reason: String,
}

impl Importance {
    /// Value per token — the density the Moirai pack by.
    #[must_use]
    pub fn density(&self, tokens: usize) -> f32 {
        self.value / tokens.max(1) as f32
    }
}

/// Weighs messages. Stateless and deterministic.
#[derive(Clone, Copy, Debug, Default)]
pub struct Themis;

impl Themis {
    /// Build the scorer.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Score one message.
    #[must_use]
    pub fn score(&self, message: &Message) -> Importance {
        let content = message.content.trim();
        // Normalized once for matching: single-spaced lowercase. (Snippets
        // are still cut from the original content.)
        let lower = content
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
            .to_lowercase();

        // Empty messages are worth nothing.
        if content.is_empty() {
            return Importance {
                value: 0.0,
                reason: "empty".to_string(),
            };
        }

        // Small talk: short and made of greetings/acks.
        if content.chars().count() <= 48 && is_small_talk(&lower) {
            return Importance {
                value: 0.12,
                reason: "small talk".to_string(),
            };
        }

        let mut value = 0.15f32;
        let mut reason: Option<String> = None;

        // Constraints carry the most weight: they change what a correct
        // answer looks like turns later.
        let mut constraint_hits = 0u32;
        let mut first_hint: Option<&str> = None;
        for hint in CONSTRAINT_HINTS {
            if contains_hint(&lower, hint) {
                constraint_hits += 1;
                if first_hint.is_none() {
                    first_hint = Some(hint);
                }
            }
        }
        if constraint_hits > 0 {
            value += 0.45 + 0.08 * (constraint_hits.min(4) - 1) as f32;
            let snippet = snippet_around(content, first_hint.unwrap_or(""));
            reason = Some(format!("contains user constraint: '{snippet}'"));
        }

        // Decisions are nearly as load-bearing as constraints.
        if let Some(hit) = DECISION_HINTS.iter().find(|h| contains_hint(&lower, h)) {
            value = value.max(0.72) + 0.05;
            if reason.is_none() {
                let snippet = snippet_around(content, hit);
                reason = Some(format!("records a decision: '{snippet}'"));
            }
        }

        // Problem reports are why support conversations exist. They outrank
        // chatter but not explicit constraints or decisions.
        if let Some(hit) = PROBLEM_HINTS.iter().find(|h| contains_hint(&lower, h)) {
            value = value.max(0.55) + 0.05;
            if reason.is_none() {
                let snippet = snippet_around(content, hit);
                reason = Some(format!("reports a problem: '{snippet}'"));
            }
        }

        // Specifics — names, amounts, quantities — are what forgetting
        // actually destroys. Cheap to detect, expensive to lose.
        let entities = count_entities(content);
        if entities > 0 {
            value += 0.06 * entities.min(4) as f32;
            if reason.is_none() {
                reason = Some("names specific people, places, or amounts".to_string());
            }
        }
        if lower.contains('@') && lower.contains('.') {
            value += 0.08; // looks like an email address
            if reason.is_none() {
                reason = Some("contains contact details".to_string());
            }
        }
        if content.chars().any(|c| c.is_ascii_digit()) {
            value += 0.08;
            if reason.is_none() {
                reason = Some("contains numbers or quantities".to_string());
            }
        }

        // Role priors: instructions frame everything; the user's words bind.
        match message.role {
            Role::System => {
                value += 0.15;
                if reason.is_none() {
                    reason = Some("system instruction".to_string());
                }
            }
            Role::User => {
                value += 0.05;
                if content.ends_with('?') {
                    value += 0.05;
                }
            }
            Role::Assistant | Role::Tool => {}
        }

        // Long, information-dense turns get a small bonus; tiny acks sink.
        let len = content.chars().count();
        if len > 200 {
            value += 0.05;
        } else if len < 20 {
            value -= 0.05;
        }

        let value = value.clamp(0.02, 0.97);
        let reason = reason.unwrap_or_else(|| "general chatter".to_string());
        Importance { value, reason }
    }

    /// Score a whole history, preserving order.
    #[must_use]
    pub fn score_all(&self, history: &[Message]) -> Vec<Importance> {
        history.iter().map(|m| self.score(m)).collect()
    }
}

/// True when `hint` occurs in `text` as whole words, not as a substring of a
/// longer word: "my" matches "my order" but not "enemy", "must" matches
/// "must go" but not "mustard". Either end of the hint may waive the boundary
/// by starting/ending with a non-alphanumeric (e.g. "$" matches "$500").
/// Both sides are expected normalized (single-spaced lowercase).
fn contains_hint(text: &str, hint: &str) -> bool {
    if hint.is_empty() {
        return false;
    }
    let left_needs_boundary = hint.chars().next().is_some_and(|c| c.is_alphanumeric());
    let right_needs_boundary = hint
        .chars()
        .next_back()
        .is_some_and(|c| c.is_alphanumeric());
    text.match_indices(hint).any(|(i, _)| {
        let left_ok = !left_needs_boundary
            || text[..i]
                .chars()
                .next_back()
                .is_none_or(|c| !c.is_alphanumeric());
        let right_ok = !right_needs_boundary
            || text[i + hint.len()..]
                .chars()
                .next()
                .is_none_or(|c| !c.is_alphanumeric());
        left_ok && right_ok
    })
}

/// True when a short lowercased message is essentially all small talk.
fn is_small_talk(lower: &str) -> bool {
    // Strip punctuation, then check the whole thing is small-talk phrases
    // glued together (covers "hi!", "ok thanks", "cool, sounds good").
    let stripped: String = lower
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c.is_whitespace() {
                c
            } else {
                ' '
            }
        })
        .collect();
    let words: Vec<&str> = stripped.split_whitespace().collect();
    if words.is_empty() {
        return true;
    }
    let joined = words.join(" ");
    if SMALL_TALK.iter().any(|p| *p == joined) {
        return true;
    }
    // Every word belongs to some small-talk phrase.
    words.iter().all(|w| {
        SMALL_TALK
            .iter()
            .any(|p| p.split_whitespace().any(|pw| pw == *w))
    })
}

/// Short evidence excerpt centered on the first match of `hint`.
fn snippet_around(content: &str, hint: &str) -> String {
    const MAX: usize = 42;
    let trimmed = content.trim().replace(['\n', '\r', '\t'], " ");
    // Collapse runs of whitespace.
    let mut collapsed = String::with_capacity(trimmed.len());
    let mut prev_space = false;
    for c in trimmed.chars() {
        if c.is_whitespace() {
            if !prev_space {
                collapsed.push(' ');
            }
            prev_space = true;
        } else {
            collapsed.push(c);
            prev_space = false;
        }
    }
    let chars: Vec<char> = collapsed.chars().collect();
    if chars.len() <= MAX {
        return collapsed;
    }
    let lower: String = collapsed.to_lowercase();
    let start = lower.find(&hint.to_lowercase()).unwrap_or(0);
    // Expand to word boundaries.
    let mut s = start;
    while s > 0 && !chars[s - 1].is_whitespace() {
        s -= 1;
    }
    let mut window: String = chars[s..].iter().collect();
    let mut wchars: Vec<char> = window.chars().collect();
    if wchars.len() > MAX {
        wchars.truncate(MAX);
        // Back off to the last word boundary.
        if let Some(last_space) = wchars.iter().rposition(|c| c.is_whitespace()) {
            wchars.truncate(last_space);
        }
        window = wchars.iter().collect::<String>() + "…";
    }
    window
}

/// Crude named-entity count: capitalized words of length ≥ 2 that are not the
/// first word of the message (so sentence case doesn't inflate the score).
fn count_entities(content: &str) -> u32 {
    let mut count = 0u32;
    for (i, word) in content.split_whitespace().enumerate() {
        let word: String = word.chars().filter(|c| c.is_alphanumeric()).collect();
        if word.len() < 2 || i == 0 {
            continue;
        }
        let mut chars = word.chars();
        let first = chars.next().unwrap_or(' ');
        if first.is_uppercase() && chars.any(|c| c.is_lowercase()) {
            count += 1;
        }
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constraint_beats_small_talk() {
        let themis = Themis::new();
        let fact = Message::user("msg_001", "Remember: my budget is under $500.");
        let chat = Message::assistant("msg_002", "ok thanks!");
        assert!(themis.score(&fact).value > 0.7);
        assert!(themis.score(&chat).value < 0.3);
    }

    #[test]
    fn hints_match_whole_words_only() {
        let themis = Themis::new();
        // Substring traps: each contains a hint's letters but not the word.
        for text in [
            "the enemy approached at dawn",    // not "my"
            "add mustard to the grocery list", // not "must"
            "I was following the river path",  // not "follow"
        ] {
            let importance = themis.score(&Message::user("msg_x", text));
            assert!(
                importance.value < 0.4,
                "{text:?} should not score as a constraint: {importance:?}"
            );
        }
        // Controls: the real words still fire.
        assert!(
            themis
                .score(&Message::user("msg_y", "my order number is 4471-B"))
                .value
                > 0.7
        );
        assert!(
            themis
                .score(&Message::user(
                    "msg_z",
                    "we must leave right now, the store closes soon"
                ))
                .value
                > 0.6
        );
    }

    #[test]
    fn problem_reports_and_preferences_score_high() {
        let themis = Themis::new();
        let damage = Message::user(
            "msg_003",
            "it arrived with a cracked screen, the box looked fine outside",
        );
        let preference = Message::user(
            "msg_019",
            "I'd like a replacement, not a refund — the refund takes too long",
        );
        let tangent = Message::user(
            "msg_008",
            "by the way, the last invoice seemed off, there was a strange charge",
        );
        assert!(
            themis.score(&damage).value >= 0.55,
            "damage report should outrank chatter"
        );
        assert!(
            themis.score(&preference).value > 0.7,
            "stated preference should score high"
        );
        assert!(
            themis.score(&tangent).value < 0.4,
            "off-topic tangent should score low"
        );
    }
}
