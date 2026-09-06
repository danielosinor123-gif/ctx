# CTX

**Your AI forgot something important. It didn't tell you. CTX would have.**

Every long conversation with an LLM eventually hits the same wall: the context window fills up, and something has to go. Almost every framework handles this the same lazy way — chop the oldest messages off the front and hope nothing important was in there.

Sometimes something important was in there. The user's name. A decision they made three turns ago. The one constraint that made their question make sense. It's gone, silently, and the first time anyone finds out is when the bot says something confidently wrong.

CTX is a context-compaction engine with one job: **decide what to keep, decide what to cut, and never do it silently.**

Every mechanism in CTX is named after the god whose myth actually describes what it does. Not decoration — a map:

> *Chronos sets the limit no one can cross. Themis weighs what each message is worth. The Moirai decide what survives on merit, not age. What doesn't survive either falls into Lethe and is gone, or is gathered by Mnemosyne into something smaller but still true. Clio writes down everything that happened, so nothing is lost without record.*

---

## The villain

**Naive truncation.** Drop-the-oldest-messages is the default in almost every chat pipeline shipping today, because it's five lines of code and nobody budgets time to do better. It works fine — right up until it doesn't, and by then the damage is a support ticket, not a stack trace.

CTX exists because "the model forgot" should never be an acceptable answer to "why did this break."

## The receipt — kept by Clio

Muse of history. Her only job is to record what happened and why, so nothing is lost to time without a trace. Every time CTX drops, keeps, or compresses a message, Clio writes it down.

```
DROPPED     msg_004      "importance: 0.12 (small talk) — budget exceeded by 340 tokens"
KEPT        msg_009      "importance: 0.91 (contains user constraint: 'budget under $500')"
SUMMARIZED  msgs_001..003 "compressed to 41 tokens, key facts preserved: [name, goal]"
```

No more guessing what your context window ate. You get a diffable, inspectable log of every decision, every time. If your bot forgets something it shouldn't have, you'll know exactly which line to blame — and whether it was actually the wrong call.

## The number

**Lethe** (naive sliding-window forgetting) vs. **the Moirai** (CTX's priority-based retention), tested against 50 synthetic long conversations where the final question depends on an early-turn fact:

| Strategy | Needle fact retained |
|---|---|
| Lethe — forgets oldest first, no judgment | 62.0% (31/50, 95% CI [48%, 74%]) |
| Moirai — decides what survives on merit | 94.0% (47/50, 95% CI [84%, 98%]) |

Same conversations. Same token budget. The only difference is what got kept.
McNemar p<0.0001 on the paired gap — and note what this is: *retention*, not
model answers. No model runs in this eval; "retained" is the honest verb.

*(Full benchmark methodology and eval set in `/benchmarks` — rerun it yourself with `cargo run --example bench`, don't take our word for it.)*

## The install

```bash
cargo add ctx
```

```rust
use ctx::{Chronos, Moirai, Compactor};

// Chronos sets the hard limit no strategy is allowed to cross
let budget = Chronos::new(4096, 512);

// The Moirai decide what survives on merit, not age
let fates = Moirai::new(budget);
let (kept_history, receipt) = fates.compact(&conversation_history)?;

// kept_history → what actually goes to the model
// receipt      → Clio's record of exactly what got dropped, kept, or
//                 summarized, and why (see `Clio` below)
```

Three lines. No server to run, no service to babysit. It's a library, not a platform — it slots into whatever pipeline you've already built. Zero dependencies, deterministic, works offline.

## The three strategies

CTX ships three, pick per use case. All three implement `Compactor` and report to Clio, so you can A/B them on your own conversations and see which one actually serves your use case — not just take a default on faith.

- **`Lethe`** — the river of forgetting. Keeps the most recent N tokens, drops the rest, oldest first. No judgment, no scoring — just fast, honest, mechanical forgetting. The baseline everyone ships by default, and the one CTX exists to improve on.

- **`Mnemosyne`** — titan of memory. Doesn't discard old turns, compresses them: old messages are folded into a short summary at sentence granularity — one load-bearing sentence can survive even when its whole turn is too diffuse to keep (bring your own model call via the `Summarizer` trait, or use the shipped dependency-free `ExtractiveSummarizer`), recent turns stay verbatim on top. Best quality-per-token when you can afford the extra call — memory kept, just smaller.

- **`Moirai`** — the Fates. Every message is weighed by **Themis** (the importance scorer — constraints, decisions, and named entities score high; small talk scores low), and the Moirai keep whichever messages are worth keeping, regardless of how old they are. This is the strategy behind the 94% number above — survival by merit, not by recency.

Two details worth knowing:

- The newest turn is always pinned (Moirai, Mnemosyne). It is the question under discussion — cutting it would be technically optimal and practically absurd. Clio marks it as pinned so the exception is on the record.
- Token counts are a deterministic heuristic (~4 chars/token + per-message overhead). Bring a real tokenizer via the `TokenCounter` trait if you want exact budgeting.

## What CTX is not

It's not a gateway. It's not a cache. It's not a guardrail system. Those are all real problems, and they're already solved well by tools like LiteLLM, Helicone, and Langfuse. CTX does one thing — deciding what survives when your context window runs out — and does it with receipts, because that's the part every other tool currently does in the dark.

## Development

```bash
cargo test                 # 34 tests: strategies, scoring, receipts, stats, model harness, edge cases
cargo run --example quickstart   # the three-line install, live
cargo run --example support_sam  # worked 24-turn support-chat scenario, all five mechanisms
cargo run --example bench        # the 50-conversation benchmark
cargo clippy -- -D warnings
cargo fmt --check
```

Layout: `src/` maps one god per module (`chronos.rs`, `themis.rs`, `moirai.rs`, `lethe.rs`, `mnemosyne.rs`, `clio.rs`) plus `message.rs`, `tokens.rs`, `strategy.rs` (the shared `Compactor` trait), and `error.rs`. Integration tests live in `tests/`, methodology in `benchmarks/README.md`.

---

**If your AI product has ever "forgotten" something a user told it, that wasn't a hallucination. That was a compaction decision nobody logged.**
