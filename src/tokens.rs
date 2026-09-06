//! Token counting.
//!
//! CTX ships with a dependency-free heuristic estimator (~4 characters per
//! token, plus per-message framing overhead). It is deliberately conservative
//! and deterministic — good enough for budgeting, and honest about being an
//! estimate.
//!
//! If you have a real tokenizer, implement [`TokenCounter`] and use
//! [`total_tokens_with`] / [`count_with`] in your own budget checks. The
//! compaction strategies themselves use the estimator so results stay
//! reproducible across machines.

use crate::Message;

/// Per-message framing overhead in tokens (role markers, separators), the way
/// chat APIs bill for message structure on top of raw text.
pub const MESSAGE_OVERHEAD: usize = 4;

/// Estimate the tokens in raw text: `ceil(chars / 4)`, minimum 1 for
/// non-empty text.
#[must_use]
pub fn estimate_tokens(text: &str) -> usize {
    if text.is_empty() {
        return 0;
    }
    text.chars().count().div_ceil(4)
}

/// Estimate the tokens for one full message: text plus framing overhead.
#[must_use]
pub fn count_message_tokens(content: &str) -> usize {
    estimate_tokens(content) + MESSAGE_OVERHEAD
}

/// Total estimated tokens for a slice of messages.
#[must_use]
pub fn total_tokens(messages: &[Message]) -> usize {
    messages.iter().map(Message::tokens).sum()
}

/// A pluggable token counter for users with a real tokenizer.
pub trait TokenCounter {
    /// Count the tokens in raw text (no framing overhead).
    fn count(&self, text: &str) -> usize;
}

/// The default heuristic counter.
#[derive(Clone, Copy, Debug, Default)]
pub struct CharTokenCounter;

impl TokenCounter for CharTokenCounter {
    fn count(&self, text: &str) -> usize {
        estimate_tokens(text)
    }
}

/// Total tokens for messages using a custom counter (plus [`MESSAGE_OVERHEAD`]
/// per message).
pub fn total_tokens_with<C: TokenCounter>(messages: &[Message], counter: &C) -> usize {
    messages
        .iter()
        .map(|m| counter.count(&m.content) + MESSAGE_OVERHEAD)
        .sum()
}

/// Count one message with a custom counter.
pub fn count_with<C: TokenCounter>(content: &str, counter: &C) -> usize {
    counter.count(content) + MESSAGE_OVERHEAD
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Message;

    #[test]
    fn estimator_scales_with_length() {
        assert_eq!(estimate_tokens(""), 0);
        assert_eq!(estimate_tokens("hi"), 1);
        assert!(estimate_tokens(&"x".repeat(400)) >= 95);
    }

    #[test]
    fn custom_counter_plumbs_through() {
        struct Words;
        impl TokenCounter for Words {
            fn count(&self, text: &str) -> usize {
                text.split_whitespace().count()
            }
        }
        let counter = CharTokenCounter;
        assert_eq!(
            counter.count("hello world, this is a test"),
            estimate_tokens("hello world, this is a test")
        );

        let words = Words;
        let history = vec![Message::user("msg_001", "one two three")];
        assert_eq!(total_tokens_with(&history, &words), 3 + MESSAGE_OVERHEAD);
        assert_eq!(
            count_with("one two three four", &words),
            4 + MESSAGE_OVERHEAD
        );
    }
}
