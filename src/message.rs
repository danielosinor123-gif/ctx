//! A single chat turn. The unit of survival.

use crate::tokens::count_message_tokens;

/// Who said it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Role {
    /// Developer / system instructions. Weighed heavily: instructions change
    /// the meaning of everything after them.
    System,
    /// The human. Constraints and decisions from the user score highest.
    User,
    /// The model.
    Assistant,
    /// Tool output.
    Tool,
}

impl Role {
    /// Short lowercase label (`"user"`, `"assistant"`, …).
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Tool => "tool",
        }
    }
}

impl std::fmt::Display for Role {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// One message in the conversation history.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Message {
    /// Stable id (`"msg_004"`). Used by Clio so every decision is traceable
    /// back to the exact message it affected.
    pub id: String,
    /// Who said it.
    pub role: Role,
    /// The text.
    pub content: String,
}

impl Message {
    /// Build a message with an explicit role.
    pub fn new(id: impl Into<String>, role: Role, content: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            role,
            content: content.into(),
        }
    }

    /// A system instruction.
    pub fn system(id: impl Into<String>, content: impl Into<String>) -> Self {
        Self::new(id, Role::System, content)
    }

    /// A user turn.
    pub fn user(id: impl Into<String>, content: impl Into<String>) -> Self {
        Self::new(id, Role::User, content)
    }

    /// An assistant turn.
    pub fn assistant(id: impl Into<String>, content: impl Into<String>) -> Self {
        Self::new(id, Role::Assistant, content)
    }

    /// A tool-output turn.
    pub fn tool(id: impl Into<String>, content: impl Into<String>) -> Self {
        Self::new(id, Role::Tool, content)
    }

    /// Estimated size of this message in tokens, including a small per-message
    /// overhead (role markers, framing) the way chat APIs charge for it.
    #[must_use]
    pub fn tokens(&self) -> usize {
        count_message_tokens(&self.content)
    }
}
