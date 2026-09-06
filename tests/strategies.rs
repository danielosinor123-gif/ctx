//! Integration tests: the myths, held to their promises.

use ctx::{
    Chronos, Clio, Compactor, Error, Importance, Lethe, Message, Mnemosyne, Moirai, Role, Themis,
};

fn budget() -> Chronos {
    Chronos::new(4096, 512)
}

fn smalltalk(id: &str, content: &str) -> Message {
    Message::assistant(id, content)
}

#[test]
fn chronos_budget_math() {
    let c = Chronos::new(4096, 512);
    assert_eq!(c.budget(), 3584);
    assert!(c.fits(3584));
    assert!(!c.fits(3585));
    assert_eq!(c.over_by(3924), 340);
    assert_eq!(Chronos::new(100, 200).budget(), 0);
}

#[test]
fn themis_weighs_merit_not_age() {
    let themis = Themis::new();
    let fact: Importance = themis.score(&Message::user(
        "msg_001",
        "Remember: my budget is under $500.",
    ));
    let chat: Importance = themis.score(&smalltalk("msg_002", "ok thanks!"));
    assert!(fact.value > 0.7, "constraint should score high: {fact:?}");
    assert!(chat.value < 0.3, "small talk should score low: {chat:?}");
    assert!(fact.reason.contains("constraint"));
}

#[test]
fn lethe_keeps_newest_drops_oldest() {
    let history: Vec<Message> = (1..=10)
        .map(|i| {
            Message::user(
                format!("msg_{i:03}"),
                format!("filler turn number {i} with some extra words to take space"),
            )
        })
        .collect();
    let total: usize = history.iter().map(Message::tokens).sum();
    // Budget for roughly the last 3 turns.
    let keep3: usize = history[7..].iter().map(Message::tokens).sum();
    let lethe = Lethe::new(Chronos::new(keep3 + 5, 0));
    let (kept, receipt) = lethe.compact(&history).unwrap();

    assert!(kept.iter().any(|m| m.id == "msg_010"));
    assert!(!kept.iter().any(|m| m.id == "msg_001"));
    assert!(receipt.dropped().contains(&"msg_001"));
    assert!(receipt.kept().contains(&"msg_010"));
    let out: usize = kept.iter().map(Message::tokens).sum();
    assert!(out <= total);
}

#[test]
fn moirai_keeps_early_constraint_over_recent_chatter() {
    let mut history = vec![Message::user(
        "msg_001",
        "Remember: my budget is under $500 for the whole trip.",
    )];
    for i in 2..=12 {
        history.push(smalltalk(
            &format!("msg_{i:03}"),
            "hey, just chatting about random everyday stuff here",
        ));
    }
    history.push(Message::user("msg_013", "So, what should I book?"));

    let total: usize = history.iter().map(Message::tokens).sum();
    // Tight budget: nowhere near enough for everything.
    let tight = Chronos::new(120, 0);
    assert!(total > tight.budget());

    let moirai = Moirai::new(tight);
    let (kept, receipt) = moirai.compact(&history).unwrap();
    let ids: Vec<&str> = kept.iter().map(|m| m.id.as_str()).collect();

    assert!(ids.contains(&"msg_001"), "early constraint must survive");
    assert!(ids.contains(&"msg_013"), "newest turn is pinned");
    assert!(
        kept.windows(2).all(|w| w[0].id < w[1].id),
        "chronological order"
    );
    let text = format!("{receipt}");
    assert!(text.contains("KEPT"));
    assert!(text.contains("DROPPED"));
    assert!(text.contains("msg_001"));
}

#[test]
fn moirai_keeps_everything_when_it_fits() {
    let history = vec![
        Message::user("msg_001", "Remember: my budget is under $500."),
        smalltalk("msg_002", "ok thanks!"),
    ];
    let (kept, receipt) = Moirai::new(budget()).compact(&history).unwrap();
    assert_eq!(kept.len(), 2);
    assert!(receipt.dropped().is_empty());
}

#[test]
fn strategies_pin_newest_turn_even_when_it_overflows() {
    let history = vec![Message::user("msg_001", "x".repeat(2000))];
    let tiny = Chronos::new(50, 0);
    let (kept, _) = Moirai::new(tiny).compact(&history).unwrap();
    assert_eq!(kept.len(), 1);
    let (kept, _) = Lethe::new(tiny).compact(&history).unwrap();
    assert_eq!(kept.len(), 1);
}

#[test]
fn zero_budget_is_an_error() {
    let history = vec![Message::user("msg_001", "hello")];
    let zero = Chronos::new(0, 0);
    assert_eq!(
        Moirai::new(zero).compact(&history).unwrap_err(),
        Error::ZeroBudget
    );
    assert_eq!(
        Lethe::new(zero).compact(&history).unwrap_err(),
        Error::ZeroBudget
    );
}

#[test]
fn empty_history_is_fine() {
    let (kept, receipt) = Moirai::new(budget()).compact(&[]).unwrap();
    assert!(kept.is_empty());
    assert!(receipt.entries.is_empty());
}

#[test]
fn mnemosyne_summarizes_old_keeps_recent_verbatim() {
    let mut history = vec![Message::user(
        "msg_001",
        "Remember: my budget is under $500 for the whole trip.",
    )];
    for i in 2..=10 {
        history.push(smalltalk(
            &format!("msg_{i:03}"),
            "chatting about hiking trails and weekend plans outdoors",
        ));
    }
    history.push(Message::user("msg_011", "What should I book?"));

    let total: usize = history.iter().map(Message::tokens).sum();
    let tight = Chronos::new(150, 0);
    assert!(total > tight.budget());

    let mne = Mnemosyne::new(tight).with_recent_turns(2);
    let (kept, receipt) = mne.compact(&history).unwrap();

    // Recent turns verbatim on top.
    assert_eq!(kept.last().unwrap().id, "msg_011");
    assert!(kept.iter().any(|m| m.id == "msg_summary"));
    // The summary carries the fact forward in compressed form.
    let summary = kept.iter().find(|m| m.id == "msg_summary").unwrap();
    assert!(
        summary.content.contains("$500"),
        "summary must preserve the fact"
    );
    assert!(!receipt.summarized().is_empty());
    let out: usize = kept.iter().map(Message::tokens).sum();
    assert!(out <= tight.budget());
}

#[test]
fn clio_receipt_format_matches_spec() {
    let mut clio = Clio::new();
    clio.dropped(
        "msg_004",
        "importance: 0.12 (small talk) — budget exceeded by 340 tokens",
    );
    clio.kept(
        "msg_009",
        "importance: 0.91 (contains user constraint: 'budget under $500')",
    );
    clio.summarized(
        "msgs_001..003",
        "compressed to 41 tokens, key facts preserved: [name, goal]",
    );
    let text = format!("{}", clio.finish(500, 160, 160));
    for line in [
        "DROPPED",
        "msg_004",
        "KEPT",
        "msg_009",
        "SUMMARIZED",
        "msgs_001..003",
    ] {
        assert!(text.contains(line), "missing {line} in:\n{text}");
    }
}

#[test]
fn roles_and_ids_survive_compaction() {
    let history = vec![
        Message::system("msg_001", "You are a helpful travel assistant."),
        Message::user("msg_002", "Remember: my budget is under $500."),
        Message::user("msg_003", "What should I book?"),
    ];
    let (kept, _) = Moirai::new(Chronos::new(60, 0)).compact(&history).unwrap();
    assert!(kept
        .iter()
        .any(|m| m.role == Role::System || m.id == "msg_002"));
    assert_eq!(kept.last().unwrap().id, "msg_003");
}

#[test]
fn no_strategy_exceeds_chronos_limit() {
    // Long meandering history, tight budget: every strategy's output must
    // fit, including Mnemosyne's generated summary message with overhead.
    let mut history = vec![
        Message::user("msg_001", "my order number is 4471-B"),
        Message::user("msg_002", "it arrived with a cracked screen"),
    ];
    for i in 3..=24 {
        history.push(Message::assistant(
            format!("msg_{i:03}"),
            "chatting about hiking trails and weekend plans outdoors",
        ));
    }
    history.push(Message::user(
        "msg_025",
        "so what's happening with my order?",
    ));

    let budget = Chronos::new(90, 0);
    let strategies: Vec<Box<dyn Compactor>> = vec![
        Box::new(Lethe::new(budget)),
        Box::new(Moirai::new(budget)),
        Box::new(Mnemosyne::new(budget).with_recent_turns(6)),
    ];
    for strategy in &strategies {
        let (kept, receipt) = strategy.compact(&history).unwrap();
        let out: usize = kept.iter().map(Message::tokens).sum();
        assert!(
            out <= budget.budget(),
            "{} exceeded budget: {out} > {}",
            strategy.name(),
            budget.budget()
        );
        assert_eq!(receipt.output_tokens, out);
    }
}

#[test]
fn mnemosyne_sentence_compression_saves_needles_in_long_turns() {
    // The needle sentence is buried in one long diffuse turn: too big to
    // pack whole, but its best sentence must survive compression.
    let journal = "Day one began with a late arrival and soup at the cafe. \
         Day two was the lake loop at an easy pace with long rests. \
         The big lesson concerns the budget: keep the whole trip under $500 or it falls apart. \
         That ceiling shapes every choice below.";
    let mut history = vec![Message::user("msg_001", journal)];
    for i in 2..=12 {
        history.push(Message::assistant(
            format!("msg_{i:03}"),
            "chatting about hiking trails and weekend plans outdoors",
        ));
    }
    history.push(Message::user(
        "msg_013",
        "what was the key thing I asked you to remember",
    ));

    let budget = Chronos::new(150, 0);
    let mne = Mnemosyne::new(budget).with_recent_turns(2);
    let (kept, receipt) = mne.compact(&history).unwrap();

    let summary = kept
        .iter()
        .find(|m| m.id == "msg_summary")
        .expect("old turns should be summarized");
    assert!(
        summary.content.contains("$500"),
        "summary must carry the needle sentence: {}",
        summary.content
    );
    assert!(!receipt.summarized().is_empty());
    let out: usize = kept.iter().map(Message::tokens).sum();
    assert!(out <= budget.budget());
}
