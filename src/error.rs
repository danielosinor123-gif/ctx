//! What can go wrong. (Very little — compaction is total by design.)

use std::fmt;

/// A compaction failure.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    /// Chronos was given a budget of zero usable tokens. Nothing can fit;
    /// not even honesty. Raise the budget.
    ZeroBudget,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::ZeroBudget => {
                write!(f, "token budget is zero: nothing can be kept")
            }
        }
    }
}

impl std::error::Error for Error {}

/// Convenience alias used by every [`crate::Compactor`].
pub type Result<T> = std::result::Result<T, Error>;
