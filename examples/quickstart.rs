//! Quickstart: three lines to stop forgetting silently.

use ctx::{Chronos, Compactor, Message, Moirai};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let conversation_history = vec![
        Message::user(
            "msg_001",
            "Hi, I'm Amara. Remember: my budget is under $500.",
        ),
        Message::assistant(
            "msg_002",
            "Hi Amara! Noted — under $500. Where are you headed?",
        ),
        Message::user("msg_003", "ok thanks!"),
        Message::assistant("msg_004", "No problem! Let me know what you need."),
        Message::user("msg_005", "hey, just thinking out loud about weekend plans"),
        Message::assistant(
            "msg_006",
            "Sounds fun! The lake trail is lovely this time of year.",
        ),
        Message::user(
            "msg_007",
            "cool, maybe. what laptop should I buy for the trip?",
        ),
    ];

    // Chronos sets the hard limit no strategy is allowed to cross.
    // (Deliberately tight here so you can watch the Fates at work.)
    let budget = Chronos::new(70, 0);

    // The Moirai decide what survives on merit, not age.
    let fates = Moirai::new(budget);
    let (kept_history, receipt) = fates.compact(&conversation_history)?;

    // kept_history → what actually goes to the model.
    println!("--- kept history ({} messages) ---", kept_history.len());
    for msg in &kept_history {
        println!("[{}] {}: {}", msg.id, msg.role, msg.content);
    }

    // receipt → Clio's record of exactly what got dropped, kept, or
    // summarized, and why.
    println!("\n--- Clio's receipt ---");
    println!("{receipt}");
    println!("\n{}", receipt.summary());

    Ok(())
}
