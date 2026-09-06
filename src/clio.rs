//! Clio — muse of history. She writes down everything that happened.
//!
//! Every compaction run produces a [`Receipt`]: one line per message saying
//! whether it was kept, dropped, or summarized, and why. Diffable,
//! inspectable, logged every time — so when the bot forgets something it
//! shouldn't have, you know exactly which line to blame.
//!
//! ```text
//! DROPPED     msg_004      "importance: 0.12 (small talk) — budget exceeded by 340 tokens"
//! KEPT        msg_009      "importance: 0.91 (contains user constraint: 'budget under $500')"
//! SUMMARIZED  msgs_001..003 "compressed to 41 tokens, key facts preserved: [name, goal]"
//! ```

use std::fmt;

/// What happened to a message (or a range of messages).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Decision {
    /// Survived into the compacted history.
    Kept,
    /// Cut to fit the budget. Gone — check the detail for why.
    Dropped,
    /// Folded into a summary by Mnemosyne. Smaller, but still true.
    Summarized,
}

impl Decision {
    #[must_use]
    fn label(self) -> &'static str {
        match self {
            Decision::Kept => "KEPT",
            Decision::Dropped => "DROPPED",
            Decision::Summarized => "SUMMARIZED",
        }
    }
}

/// One line in Clio's record.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Entry {
    /// What happened.
    pub decision: Decision,
    /// Which message (`"msg_004"`) or range (`"msgs_001..003"`).
    pub target: String,
    /// Why, in human words. No codes, no silent cuts.
    pub detail: String,
}

/// Clio's record of one compaction run.
///
/// Build it through [`Clio`], read it back here. Displays as one fixed-width
/// line per decision, newest-last in the order decisions were recorded.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Receipt {
    /// Every decision, in the order it was recorded.
    pub entries: Vec<Entry>,
    /// Tokens in before compaction.
    pub input_tokens: usize,
    /// Tokens out after compaction.
    pub output_tokens: usize,
    /// The budget Chronos set.
    pub budget: usize,
}

impl Receipt {
    /// Ids (targets) of everything kept.
    #[must_use]
    pub fn kept(&self) -> Vec<&str> {
        self.targets(Decision::Kept)
    }

    /// Ids (targets) of everything dropped.
    #[must_use]
    pub fn dropped(&self) -> Vec<&str> {
        self.targets(Decision::Dropped)
    }

    /// Ids (targets) of everything summarized.
    #[must_use]
    pub fn summarized(&self) -> Vec<&str> {
        self.targets(Decision::Summarized)
    }

    fn targets(&self, decision: Decision) -> Vec<&str> {
        self.entries
            .iter()
            .filter(|e| e.decision == decision)
            .map(|e| e.target.as_str())
            .collect()
    }

    /// One-line summary: `kept 8/11 messages, 380/512 tokens (budget 512)`.
    #[must_use]
    pub fn summary(&self) -> String {
        let kept_msgs = self.kept().len();
        let total_msgs = self.entries.len();
        format!(
            "kept {kept_msgs}/{total_msgs} messages, {}/{} tokens (budget {})",
            self.output_tokens, self.input_tokens, self.budget
        )
    }
}

impl fmt::Display for Receipt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, entry) in self.entries.iter().enumerate() {
            if i > 0 {
                f.write_str("\n")?;
            }
            write!(
                f,
                "{:<12}{:<13}\"{}\"",
                entry.decision.label(),
                entry.target,
                entry.detail
            )?;
        }
        Ok(())
    }
}

/// The recorder. Strategies write every decision here as they make it; when
/// the run is over, [`Clio::finish`] seals the log into a [`Receipt`].
#[derive(Clone, Debug, Default)]
pub struct Clio {
    entries: Vec<Entry>,
}

impl Clio {
    /// Open a fresh record.
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Record a message that survived, and why.
    pub fn kept(&mut self, id: impl Into<String>, detail: impl Into<String>) {
        self.entries.push(Entry {
            decision: Decision::Kept,
            target: id.into(),
            detail: detail.into(),
        });
    }

    /// Record a message that was cut, and why.
    pub fn dropped(&mut self, id: impl Into<String>, detail: impl Into<String>) {
        self.entries.push(Entry {
            decision: Decision::Dropped,
            target: id.into(),
            detail: detail.into(),
        });
    }

    /// Record a range folded into a summary, and what survived of it.
    pub fn summarized(&mut self, range: impl Into<String>, detail: impl Into<String>) {
        self.entries.push(Entry {
            decision: Decision::Summarized,
            target: range.into(),
            detail: detail.into(),
        });
    }

    /// Seal the log.
    #[must_use]
    pub fn finish(self, input_tokens: usize, output_tokens: usize, budget: usize) -> Receipt {
        Receipt {
            entries: self.entries,
            input_tokens,
            output_tokens,
            budget,
        }
    }

    /// `msgs_001..003`-style range label for a compaction run over old turns.
    #[must_use]
    pub fn range_label(first_id: &str, last_id: &str) -> String {
        if first_id == last_id {
            first_id.to_string()
        } else {
            format!("{first_id}..{last_id}")
        }
    }
}
