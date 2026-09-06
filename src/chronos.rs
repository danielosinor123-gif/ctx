//! Chronos — the limit no one can cross.
//!
//! Chronos is the hard token budget. Every strategy receives one and no
//! strategy is allowed to return history that exceeds it (except the single
//! degenerate case where even the newest message alone overflows, which is
//! kept anyway and flagged in the receipt — dropping the user's latest turn
//! silently would be worse).

use crate::tokens::total_tokens;
use crate::Message;

/// The hard token limit for a compaction run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Chronos {
    max_tokens: usize,
    reserved_for_output: usize,
}

impl Chronos {
    /// Define the limit.
    ///
    /// * `max_tokens` — the model's context window.
    /// * `reserved_for_output` — tokens held back for the reply. The usable
    ///   budget for history is `max_tokens - reserved_for_output`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use ctx::Chronos;
    /// let budget = Chronos::new(4096, 512);
    /// assert_eq!(budget.budget(), 3584);
    /// ```
    #[must_use]
    pub fn new(max_tokens: usize, reserved_for_output: usize) -> Self {
        Self {
            max_tokens,
            reserved_for_output,
        }
    }

    /// The full context window.
    #[must_use]
    pub fn max_tokens(self) -> usize {
        self.max_tokens
    }

    /// Tokens held back for generation.
    #[must_use]
    pub fn reserved_for_output(self) -> usize {
        self.reserved_for_output
    }

    /// Usable budget for history: `max_tokens - reserved_for_output`
    /// (saturating, never negative).
    #[must_use]
    pub fn budget(self) -> usize {
        self.max_tokens.saturating_sub(self.reserved_for_output)
    }

    /// Does `history_tokens` fit inside the budget?
    #[must_use]
    pub fn fits(self, history_tokens: usize) -> bool {
        history_tokens <= self.budget()
    }

    /// How many tokens over budget is `history_tokens`? Zero when it fits.
    #[must_use]
    pub fn over_by(self, history_tokens: usize) -> usize {
        history_tokens.saturating_sub(self.budget())
    }

    /// Token count of a history slice (estimator, includes framing overhead).
    #[must_use]
    pub fn weigh(self, history: &[Message]) -> usize {
        total_tokens(history)
    }
}
