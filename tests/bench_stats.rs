//! Tests for the benchmark's dependency-free statistics.
//!
//! The helpers live in the eval harness (`examples/bench.rs`), not the
//! library — they are statistics about the experiment, not compaction
//! machinery. Included here by path so `cargo test` verifies them.

#[allow(dead_code)] // this module is the whole harness; tests use only its stats
#[path = "../examples/bench.rs"]
mod bench;

use bench::{erf, mcnemar, normal_cdf, wilson};

#[test]
fn erf_matches_known_values() {
    assert!((erf(0.0)).abs() < 1e-12);
    assert!((erf(1.0) - 0.8427007929).abs() < 1e-6);
    assert!((erf(-1.0) + 0.8427007929).abs() < 1e-6);
    assert!((erf(3.0) - 0.9999779095).abs() < 1e-6);
}

#[test]
fn normal_cdf_matches_known_values() {
    assert!((normal_cdf(0.0) - 0.5).abs() < 1e-12);
    assert!((normal_cdf(1.96) - 0.9750021049).abs() < 1e-6);
    assert!((normal_cdf(-1.96) - 0.0249978951).abs() < 1e-6);
}

#[test]
fn wilson_brackets_known_intervals() {
    // 31/50 ≈ 62%: published interval is roughly [48%, 74%].
    let (lo, hi) = wilson(31.0 / 50.0, 50);
    assert!(lo > 0.47 && lo < 0.50, "lo={lo}");
    assert!(hi > 0.73 && hi < 0.76, "hi={hi}");
    // 50/50: lower bound stays below 1, upper bound pins at 1.
    let (lo, hi) = wilson(1.0, 50);
    assert!(lo > 0.90 && lo < 1.0, "lo={lo}");
    assert_eq!(hi, 1.0);
    // Interval always contains the point estimate and stays in [0, 1].
    for (k, n) in [(0, 50), (1, 50), (25, 50), (47, 50)] {
        let (lo, hi) = wilson(k as f64 / n as f64, n);
        assert!(lo <= k as f64 / n as f64 && k as f64 / n as f64 <= hi);
        assert!(lo >= 0.0 && hi <= 1.0);
    }
}

#[test]
fn mcnemar_matches_known_values() {
    // Total agreement: chi2 0, p 1.
    assert_eq!(mcnemar(0, 0), (0.0, 1.0));
    // 16-to-0 split (lethe vs moirai shape): chi2 = 15^2/16, p ≈ 0.0002.
    let (chi2, p) = mcnemar(16, 0);
    assert!((chi2 - 14.0625).abs() < 1e-9, "chi2={chi2}");
    assert!(p < 0.001, "p={p}");
    assert!(p > 0.00005, "p={p}");
    // 3-to-0 split (moirai vs mnemosyne shape): real but not significant.
    let (chi2, p) = mcnemar(3, 0);
    assert!((chi2 - 4.0 / 3.0).abs() < 1e-9, "chi2={chi2}");
    assert!(p > 0.05, "p={p}");
    // Symmetric splits carry no signal (continuity correction leaves a
    // trace chi2 of 1/(b+c), so p lands near 0.8, not exactly 1).
    let (chi2, p) = mcnemar(8, 8);
    assert!((chi2 - 1.0 / 16.0).abs() < 1e-12, "chi2={chi2}");
    assert!(p > 0.79 && p < 0.81, "p={p}");
}
