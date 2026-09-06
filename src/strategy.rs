//! The contract every strategy honors.

use crate::{Message, Receipt, Result};

/// A context-compaction strategy.
///
/// Implementors take the full conversation history and return the slice of it
/// that actually goes to the model, plus Clio's [`Receipt`] recording every
/// decision. All three shipped strategies — [`crate::Lethe`],
/// [`crate::Mnemosyne`], [`crate::Moirai`] — implement this, so you can A/B
/// them on your own conversations behind one interface.
pub trait Compactor {
    /// Short name for logs and benchmark tables (`"lethe"`, …).
    fn name(&self) -> &'static str;

    /// Compact `history` down to Chronos's budget.
    ///
    /// The returned messages keep their original chronological order and
    /// their original ids. The receipt explains every keep, drop, and
    /// summarization.
    fn compact(&self, history: &[Message]) -> Result<(Vec<Message>, Receipt)>;
}
