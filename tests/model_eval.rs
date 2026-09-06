//! Tests for the real-model eval harness: arg parsing, answer graders,
//! JSON helpers, cost math, and divergence classification.
//!
//! Like `bench_stats.rs`, the code under test lives in the eval harness
//! (`examples/bench.rs`), not the library.

#[allow(dead_code)] // this module is the whole harness; tests use only parts
#[path = "../examples/bench.rs"]
mod bench;

use bench::{
    answer_tokens, call_cost, classify, find_json_string, find_json_u64, fold_token,
    grade_l1_answer, grade_l2_answer, json_escape, parse_args_from, token_matches, Divergence,
    ModelOpts,
};

fn args(words: &[&str]) -> Vec<String> {
    words.iter().map(|w| w.to_string()).collect()
}

#[test]
fn cli_defaults_to_no_model_mode() {
    assert_eq!(
        parse_args_from(&args(&[])),
        Ok(ModelOpts {
            enabled: false,
            limit: usize::MAX,
            pause_ms: 1000,
        })
    );
}

#[test]
fn cli_parses_model_options() {
    assert_eq!(
        parse_args_from(&args(&["--with-model"])),
        Ok(ModelOpts {
            enabled: true,
            limit: usize::MAX,
            pause_ms: 1000,
        })
    );
    assert_eq!(
        parse_args_from(&args(&[
            "--with-model",
            "--model-limit",
            "6",
            "--model-pause-ms",
            "0"
        ])),
        Ok(ModelOpts {
            enabled: true,
            limit: 6,
            pause_ms: 0,
        })
    );
}

#[test]
fn cli_rejects_bad_input() {
    assert!(parse_args_from(&args(&["--bogus"])).is_err());
    assert!(parse_args_from(&args(&["--model-limit"])).is_err());
    assert!(parse_args_from(&args(&["--model-limit", "0"])).is_err());
    assert!(parse_args_from(&args(&["--model-limit", "abc"])).is_err());
    // Model-scoped flags without the mode flag: refuse, don't ignore.
    assert!(parse_args_from(&args(&["--model-limit", "6"])).is_err());
}

#[test]
fn l1_is_case_insensitive_substring() {
    assert!(grade_l1_answer("Your budget is under $500.", "$500"));
    assert!(grade_l1_answer("BUDGET UNDER $500", "$500"));
    assert!(!grade_l1_answer("I don't have that information.", "$500"));
    // Known L1 weakness, kept as the strict baseline: longer numbers match.
    assert!(grade_l1_answer("Your budget is $5000.", "$500"));
}

#[test]
fn token_folding_and_matching() {
    assert_eq!(fold_token("$500"), "500");
    assert_eq!(fold_token("trains!"), "trains");
    // Hyphenated compounds split into runs (never merged).
    assert_eq!(
        answer_tokens("Carry your epi-pen."),
        vec!["carry", "your", "epi", "pen"]
    );
    assert_eq!(answer_tokens("$500, ok?"), vec!["500", "ok"]);
    assert!(token_matches("peanuts", "peanut")); // plural tolerance
    assert!(token_matches("budgets", "budget"));
    assert!(token_matches("500", "500"));
    assert!(!token_matches("5000", "500")); // L1's blind spot, fixed here
    assert!(!token_matches("open", "pen"));
    assert!(!token_matches("happen", "pen"));
    assert!(!token_matches("spend", "pen"));
}

#[test]
fn l2_tolerates_rephrasing_but_requires_digits() {
    let essentials = &["500", "budget"];
    assert!(grade_l2_answer("Your budget is under $500.", essentials));
    assert!(grade_l2_answer(
        "Budget: 500 dollars for the trip",
        essentials
    ));
    assert!(grade_l2_answer(
        "staying under budget at $500 total",
        essentials
    ));
    assert!(!grade_l2_answer(
        "I don't have that information.",
        essentials
    ));
    assert!(!grade_l2_answer("Your budget is on track.", essentials)); // no amount
                                                                       // Documented L2 gap (LLM-judge territory): spelled-out numbers fail.
    assert!(!grade_l2_answer(
        "five hundred dollars, over budget",
        essentials
    ));
    // All five eval rubrics pass on their own needle facts.
    assert!(grade_l2_answer(
        "Keep in mind: I am allergic to peanuts, so avoid them.",
        &["peanut"]
    ));
    assert!(grade_l2_answer(
        "You are Amara and the deadline is Friday.",
        &["amara", "friday"]
    ));
    assert!(grade_l2_answer("Take the train instead.", &["train"]));
    assert!(grade_l2_answer("Carry your epi-pen.", &["epi", "pen"]));
}

#[test]
fn divergence_covers_all_four_cells() {
    use Divergence::{BothRight, BothWrong, DroppedButRight, RetainedButWrong};
    assert_eq!(classify(true, true), BothRight);
    assert_eq!(classify(true, false), RetainedButWrong);
    assert_eq!(classify(false, true), DroppedButRight);
    assert_eq!(classify(false, false), BothWrong);
}

#[test]
fn cost_math_is_per_million() {
    // 300 prompt + 50 completion tokens at 0.59/0.79 per 1M.
    let cost = call_cost(300, 50, 0.59, 0.79);
    assert!((cost - (300.0 / 1e6 * 0.59 + 50.0 / 1e6 * 0.79)).abs() < 1e-12);
    assert_eq!(call_cost(0, 0, 0.59, 0.79), 0.0);
}

#[test]
fn json_escape_covers_controls() {
    assert_eq!(json_escape("a\"b\\c"), "a\\\"b\\\\c");
    assert_eq!(json_escape("line1\nline2\ttab"), "line1\\nline2\\ttab");
    assert_eq!(json_escape("café — ok"), "café — ok"); // UTF-8 passes through
    assert_eq!(json_escape("a\x01b"), "a\\u0001b");
}

#[test]
fn json_finders_parse_response_shapes() {
    let body = r#"{"id":"x","choices":[{"message":{"role":"assistant","content":"Your budget is under $500.\nNice trip!"}}}],"usage":{"prompt_tokens":231,"completion_tokens":17,"total_tokens":248}}"#;
    assert_eq!(
        find_json_string(body, "content"),
        Some("Your budget is under $500.\nNice trip!".to_string())
    );
    assert_eq!(find_json_u64(body, "prompt_tokens"), Some(231));
    assert_eq!(find_json_u64(body, "completion_tokens"), Some(17));
    assert_eq!(find_json_u64(body, "missing"), None);
    // Error envelopes surface as messages, not content.
    let err = r#"{"error":{"message":"Invalid API Key","code":"invalid_api_key"}}"#;
    assert_eq!(find_json_string(err, "content"), None);
    assert_eq!(
        find_json_string(err, "message"),
        Some("Invalid API Key".to_string())
    );
    // Escapes and unicode survive the round trip.
    let esc = r#"{"content":"a\"b\\c\n\u00e9"}"#;
    assert_eq!(
        find_json_string(esc, "content"),
        Some("a\"b\\c\né".to_string())
    );
}
