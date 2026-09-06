//! Lethe — the river of forgetting.
//!
//! Keeps the most recent tokens that fit, drops the rest oldest-first. No
//! judgment, no scoring — fast, honest, mechanical forgetting. This is the
//! baseline almost every chat pipeline ships by default, and the villain CTX
//! exists to beat. It still reports every cut to Clio, so even the naive path
//! is never silent.

use crate::{Chronos, Clio, Compactor, Error, Message, Receipt, Result};

/// Naive sliding-window forgetting: keep the newest suffix that fits.
#[derive(Clone, Copy, Debug)]
pub struct Lethe {
    budget: Chronos,
}

impl Lethe {
    /// Build the strategy around Chronos's limit.
    #[must_use]
    pub fn new(budget: Chronos) -> Self {
        Self { budget }
    }

    /// The budget this strategy compacts down to.
    #[must_use]
    pub fn budget(self) -> Chronos {
        self.budget
    }
}

impl Compactor for Lethe {
    fn name(&self) -> &'static str {
        "lethe"
    }

    fn compact(&self, history: &[Message]) -> Result<(Vec<Message>, Receipt)> {
        let budget = self.budget.budget();
        if budget == 0 && !history.is_empty() {
            return Err(Error::ZeroBudget);
        }

        let input_tokens: usize = history.iter().map(Message::tokens).sum();
        let over_by = input_tokens.saturating_sub(budget);

        // Walk back from the newest turn, keeping everything that fits.
        // The newest message is always kept — dropping the turn the user
        // just wrote would be worse than going slightly over budget, and
        // Clio will flag the overflow.
        let mut keep_from = history.len();
        let mut kept_tokens = 0usize;
        for (i, msg) in history.iter().enumerate().rev() {
            let cost = msg.tokens();
            if kept_tokens + cost <= budget {
                keep_from = i;
                kept_tokens += cost;
            } else if keep_from == history.len() {
                // Nothing kept yet and even this newest message overflows:
                // keep it anyway and flag it.
                keep_from = i;
                kept_tokens += cost;
            } else {
                break;
            }
        }

        let mut clio = Clio::new();
        for msg in &history[..keep_from] {
            clio.dropped(
                msg.id.clone(),
                format!("sliding window cut — budget exceeded by {over_by} tokens"),
            );
        }
        for msg in &history[keep_from..] {
            let overflow_note = if kept_tokens > budget {
                format!(
                    " (over budget by {} tokens — newest turn pinned)",
                    kept_tokens - budget
                )
            } else {
                String::new()
            };
            clio.kept(
                msg.id.clone(),
                format!(
                    "within recent window, {} tokens{overflow_note}",
                    msg.tokens()
                ),
            );
        }

        let kept: Vec<Message> = history[keep_from..].to_vec();
        let output_tokens: usize = kept.iter().map(Message::tokens).sum();
        Ok((kept, clio.finish(input_tokens, output_tokens, budget)))
    }
}
