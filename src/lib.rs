//! CTX — a context-compaction engine with receipts.
//!
//! Every long conversation with an LLM eventually hits the same wall: the
//! context window fills up, and something has to go. CTX decides what to
//! keep, decides what to cut, and never does it silently.
//!
//! > *Chronos sets the limit no one can cross. Themis weighs what each
//! > message is worth. The Moirai decide what survives on merit, not age.
//! > What doesn't survive either falls into Lethe and is gone, or is gathered
//! > by Mnemosyne into something smaller but still true. Clio writes down
//! > everything that happened, so nothing is lost without record.*
//!
//! # Quickstart
//!
//! ```rust
//! use ctx::{Chronos, Moirai, Compactor};
//!
//! let conversation = vec![
//!     ctx::Message::user("msg_001", "Remember: my budget is under $500."),
//!     ctx::Message::assistant("msg_002", "Got it, I'll keep that in mind."),
//!     ctx::Message::user("msg_003", "What laptop should I buy?"),
//! ];
//!
//! // Chronos sets the hard limit no strategy is allowed to cross.
//! let budget = Chronos::new(4096, 512);
//!
//! // The Moirai decide what survives on merit, not age.
//! let fates = Moirai::new(budget);
//! let (kept_history, receipt) = fates.compact(&conversation).unwrap();
//!
//! // kept_history → what actually goes to the model.
//! // receipt      → Clio's record of what got dropped, kept, or summarized, and why.
//! println!("{receipt}");
//! # let _ = kept_history;
//! ```
//!
//! # Strategies
//!
//! All three implement [`Compactor`] and all three report to Clio, so you can
//! A/B them on your own conversations:
//!
//! - [`Lethe`] — keeps the most recent tokens, drops the rest, oldest first.
//!   No judgment, no scoring. The baseline everyone ships by default.
//! - [`Mnemosyne`] — compresses old turns into a short summary, keeps recent
//!   turns verbatim. Best quality-per-token when you can afford the summarizer.
//! - [`Moirai`] — weighs every message with [`Themis`] and keeps whichever
//!   messages are worth keeping, regardless of age. Survival by merit.

mod chronos;
mod clio;
mod error;
mod lethe;
mod message;
mod mnemosyne;
mod moirai;
mod strategy;
mod themis;
mod tokens;

pub use chronos::Chronos;
pub use clio::{Clio, Decision, Entry, Receipt};
pub use error::{Error, Result};
pub use lethe::Lethe;
pub use message::{Message, Role};
pub use mnemosyne::{ExtractiveSummarizer, Mnemosyne, Summarizer, Summary};
pub use moirai::Moirai;
pub use strategy::Compactor;
pub use themis::{Importance, Themis};
pub use tokens::{
    count_with, estimate_tokens, total_tokens, total_tokens_with, CharTokenCounter, TokenCounter,
};
