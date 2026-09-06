//! Support Sam — every CTX mechanism doing real work in one scenario.
//!
//! A customer-support chat runs 24 turns: the user states an order number
//! (turn 2), reports a cracked screen (turn 3), drifts into a billing tangent
//! (turns 8–14), confirms they want a replacement, not a refund (turn 19),
//! then asks "so what's happening with my order?" (turn 24).
//!
//! Run it: `cargo run --example support_sam`
//!
//! Notes on honesty:
//! - The token budget is deliberately tiny (90 tokens) so 24 short messages
//!   exert the same pressure a 4k window feels in production. The mechanism
//!   is identical; only the scale is shrunk to fit a demo.
//! - The "model backend" is simulated: [`answer_from`] can only see the
//!   compacted history, exactly like a real model call. If a fact didn't
//!   survive compaction, the answer can't contain it — that constraint is
//!   the whole point of the demo.

use ctx::{Chronos, Compactor, Lethe, Message, Mnemosyne, Moirai, Themis};

/// Facts turn 24 depends on. Each keyword appears ONLY in its needle turn
/// (plus deliberate assistant restatements noted below — the receipt shows
/// exactly which copy survived).
const CRITICAL_FACTS: &[(&str, &str)] = &[
    ("4471-b", "order number (turn 2)"),
    ("crack", "cracked screen (turn 3)"),
    ("not a refund", "replacement preference (turn 19)"),
];

fn conversation() -> Vec<Message> {
    vec![
        Message::user(
            "msg_001",
            "hi, I need help with an order that arrived yesterday",
        ),
        Message::user("msg_002", "my order number is 4471-B"),
        Message::user(
            "msg_003",
            "it arrived with a cracked screen, the box looked fine outside",
        ),
        Message::assistant(
            "msg_004",
            "sorry to hear that — a cracked screen on order 4471-B. let me look into your options",
        ),
        Message::user("msg_005", "thanks"),
        Message::assistant(
            "msg_006",
            "of course! just to confirm: it powers on but the glass is cracked, right",
        ),
        Message::user(
            "msg_007",
            "yeah it turns on, the crack runs across the corner",
        ),
        Message::user(
            "msg_008",
            "by the way, the last invoice seemed a bit off, there was a strange charge",
        ),
        Message::assistant(
            "msg_009",
            "I can check the invoice after we sort this out — what was the charge",
        ),
        Message::user(
            "msg_010",
            "it was a small delivery fee, maybe it is fine actually",
        ),
        Message::assistant(
            "msg_011",
            "delivery fees apply to smaller orders, that charge looks normal",
        ),
        Message::user("msg_012", "ok, and do you price match other stores"),
        Message::assistant(
            "msg_013",
            "we match prices within two weeks of purchase with a link to the listing",
        ),
        Message::user("msg_014", "ok, forget the invoice stuff then"),
        Message::assistant(
            "msg_015",
            "no problem. for the cracked screen, I can offer a replacement or a refund",
        ),
        Message::user("msg_016", "hmm, let me think for a sec"),
        Message::assistant(
            "msg_017",
            "take your time — just let me know which you prefer",
        ),
        Message::user("msg_018", "ok thanks for waiting"),
        Message::user(
            "msg_019",
            "I'd like a replacement, not a refund — the refund takes too long",
        ),
        Message::assistant(
            "msg_020",
            "a replacement it is. I'll get that set up for you now",
        ),
        Message::user("msg_021", "cool"),
        Message::assistant("msg_022", "anything else I can help with today"),
        Message::user("msg_023", "nope, all good"),
        Message::user("msg_024", "so what's happening with my order?"),
    ]
}

/// Simulated model backend: answers using ONLY what survived compaction.
/// A real backend (`backend.complete(system_prompt, &compacted)`) would see
/// exactly the same input — so whatever this function can't say, the model
/// couldn't say either.
fn answer_from(kept: &[Message]) -> String {
    let has = |needle: &str| {
        kept.iter()
            .any(|m| m.content.to_lowercase().contains(needle))
    };
    let order = has("4471-b");
    let damage = has("crack");
    let preference = has("not a refund");
    let replacement_mentioned = has("replacement");

    let mut answer = String::from("Here's where things stand: ");
    if preference {
        answer.push_str(
            "you asked for a replacement rather than a refund, so that's what's in motion. ",
        );
    } else if replacement_mentioned {
        answer.push_str("a replacement was discussed. ");
    }
    if order && damage {
        answer.push_str(
            "That's for order 4471-B, the one that arrived with a cracked screen. \
             Expect a shipping update within 2 business days.",
        );
    } else {
        let mut missing = Vec::new();
        if !order {
            missing.push("your order number");
        }
        if !damage {
            missing.push("what arrived damaged");
        }
        answer.push_str(&format!(
            "But I've lost the thread on {} — could you repeat it?",
            missing.join(" and ")
        ));
    }
    answer
}

/// Critical-fact check: the "0 critical facts dropped" line is computed, not
/// claimed. Returns (retained, total).
fn critical_check(kept: &[Message]) -> (usize, usize) {
    let retained = CRITICAL_FACTS
        .iter()
        .filter(|(keyword, _)| {
            kept.iter()
                .any(|m| m.content.to_lowercase().contains(*keyword))
        })
        .count();
    (retained, CRITICAL_FACTS.len())
}

fn main() {
    let history = conversation();

    // ---- Step 1 — Chronos sets the hard limit. ----
    // Production shape would be Chronos::new(4096, 512 + 180): window minus
    // output reserve minus the pre-counted system prompt. Here the window is
    // shrunk so the demo compacts; the boundary plays the same role.
    let budget = Chronos::new(90, 0);
    println!("== Step 1 — Chronos ==");
    println!("history budget: {} tokens\n", budget.budget());

    // ---- Step 2 — Themis scores every message before anything gets cut. ----
    println!("== Step 2 — Themis ==");
    let scorer = Themis::new();
    for msg in &history {
        let importance = scorer.score(msg);
        println!(
            "{}  {:.2}  [{}] {}",
            msg.id, importance.value, msg.role, importance.reason
        );
    }
    println!();

    // ---- Step 3 — Choose a strategy. All three, same budget, same history. ----
    println!("== Step 3 — strategies ==");
    let strategies: Vec<(&str, Box<dyn Compactor>)> = vec![
        ("lethe", Box::new(Lethe::new(budget))),
        ("moirai", Box::new(Moirai::new(budget))),
        (
            "mnemosyne",
            Box::new(Mnemosyne::new(budget).with_recent_turns(6)),
        ),
    ];
    for (name, strategy) in &strategies {
        let (kept, receipt) = strategy.compact(&history).expect("compaction is total");
        let ids: Vec<&str> = kept.iter().map(|m| m.id.as_str()).collect();
        let (retained, total) = critical_check(&kept);
        println!("{name}: kept [{}]", ids.join(", "));
        println!(
            "{name}: {} | critical facts: {retained}/{total} retained",
            receipt.summary()
        );
    }
    println!();

    // ---- Step 4 — Clio writes the receipt. ----
    // Moirai's full receipt, plus the TOTAL line for each strategy.
    println!("== Step 4 — Clio's receipt (moirai) ==");
    let moirai = Moirai::new(budget);
    let (_, receipt) = moirai.compact(&history).expect("compaction is total");
    println!("{receipt}");
    println!();
    for (name, strategy) in &strategies {
        let (kept, receipt) = strategy.compact(&history).expect("compaction is total");
        let (retained, total) = critical_check(&kept);
        println!(
            "TOTAL ({name}): {} | critical facts dropped: {}",
            receipt.summary(),
            total - retained
        );
    }
    println!();

    // ---- Step 5 — Wire it into the chat loop (turn 24, live). ----
    println!("== Step 5 — turn 24 ==");
    println!("User: {}", history[23].content);
    for name in ["lethe", "moirai"] {
        // Fresh history up to turn 23, then the new user turn arrives.
        let mut live: Vec<Message> = history[..23].to_vec();
        live.push(history[23].clone());
        let strategy: Box<dyn Compactor> = match name {
            "lethe" => Box::new(Lethe::new(budget)),
            _ => Box::new(Moirai::new(budget)),
        };
        let (compacted, receipt) = strategy.compact(&live).expect("compaction is total");
        println!("\n--- {name} compacts, Clio logs, backend answers from kept history ---");
        println!("{receipt}");
        println!("Bot ({name}): {}", answer_from(&compacted));
    }
}
