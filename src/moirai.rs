//! The Moirai — the Fates. Survival by merit, not by recency.
//!
//! Every message is weighed by [`Themis`](crate::Themis): constraints,
//! decisions, and named entities score high; small talk scores low. The
//! Moirai then pack the highest importance-per-token messages into Chronos's
//! budget, regardless of how old they are.
//!
//! One exception to pure merit: the newest turn is always pinned. It is the
//! question the model is about to answer — cutting it to save a few tokens
//! would be technically optimal and practically absurd. Clio marks it as
//! pinned so the exception is on the record.

use crate::{Chronos, Clio, Compactor, Error, Message, Receipt, Result, Themis};

/// Priority-based retention. The strategy behind the 94%.
#[derive(Clone, Copy, Debug)]
pub struct Moirai {
    budget: Chronos,
    scorer: Themis,
}

impl Moirai {
    /// Build the strategy around Chronos's limit, with the default scorer.
    #[must_use]
    pub fn new(budget: Chronos) -> Self {
        Self {
            budget,
            scorer: Themis::new(),
        }
    }

    /// Use a custom [`Themis`] scorer (same type today; a hook for tuned
    /// weights tomorrow).
    #[must_use]
    pub fn with_scorer(mut self, scorer: Themis) -> Self {
        self.scorer = scorer;
        self
    }

    /// The budget this strategy compacts down to.
    #[must_use]
    pub fn budget(self) -> Chronos {
        self.budget
    }
}

impl Compactor for Moirai {
    fn name(&self) -> &'static str {
        "moirai"
    }

    fn compact(&self, history: &[Message]) -> Result<(Vec<Message>, Receipt)> {
        let budget = self.budget.budget();
        if budget == 0 && !history.is_empty() {
            return Err(Error::ZeroBudget);
        }
        if history.is_empty() {
            return Ok((Vec::new(), Clio::new().finish(0, 0, budget)));
        }

        let input_tokens: usize = history.iter().map(Message::tokens).sum();
        let over_by = input_tokens.saturating_sub(budget);

        // Fast path: everything fits. Keep it all, log it all.
        if input_tokens <= budget {
            let mut clio = Clio::new();
            for msg in history {
                let importance = self.scorer.score(msg);
                clio.kept(
                    msg.id.clone(),
                    format!(
                        "importance: {:.2} ({})",
                        importance.value, importance.reason
                    ),
                );
            }
            return Ok((
                history.to_vec(),
                clio.finish(input_tokens, input_tokens, budget),
            ));
        }

        // Score everything once.
        let scored: Vec<(usize, f32, String)> = history
            .iter()
            .map(|msg| {
                let importance = self.scorer.score(msg);
                let density = importance.density(msg.tokens());
                (
                    msg.tokens(),
                    density,
                    format!(
                        "importance: {:.2} ({})",
                        importance.value, importance.reason
                    ),
                )
            })
            .collect();

        // Pin the newest turn: it is the question under discussion.
        let last = history.len() - 1;
        let mut kept_flag = vec![false; history.len()];
        kept_flag[last] = true;
        let mut kept_tokens = history[last].tokens();

        // Pack the rest by merit density (importance per token), oldest
        // first on ties so early facts win draws.
        let mut order: Vec<usize> = (0..last).collect();
        order.sort_by(|&a, &b| {
            scored[b]
                .1
                .partial_cmp(&scored[a].1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.cmp(&b))
        });
        for i in order {
            let cost = scored[i].0;
            if kept_tokens + cost <= budget {
                kept_flag[i] = true;
                kept_tokens += cost;
            }
        }

        // Emit in chronological order: the model sees a conversation, not a
        // leaderboard.
        let mut clio = Clio::new();
        let mut kept = Vec::new();
        for (i, msg) in history.iter().enumerate() {
            if kept_flag[i] {
                let mut detail = scored[i].2.clone();
                if i == last {
                    detail.push_str("; pinned: most recent turn");
                }
                clio.kept(msg.id.clone(), detail);
                kept.push(msg.clone());
            } else {
                clio.dropped(
                    msg.id.clone(),
                    format!("{} — budget exceeded by {over_by} tokens", scored[i].2),
                );
            }
        }

        let output_tokens: usize = kept.iter().map(Message::tokens).sum();
        Ok((kept, clio.finish(input_tokens, output_tokens, budget)))
    }
}
