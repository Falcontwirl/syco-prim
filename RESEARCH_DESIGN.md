# Research Design

## Hypothesis

**H1.** The benefit of the original anti-sycophancy conversational primer shrinks as more
natural, topic-related conversation sits between the primer and the evaluation question.

With S_primed(L) and S_control(L) the sycophancy rates at intervening length L:

**B(L) = S_control(L) − S_primed(L)**, and H1 predicts B(L) decreases in L.

**H0.** B(L) does not change systematically with L.

We are testing whether the *primer's incremental benefit* fades, not whether long
conversations make models more sycophantic in general (the control arm at each L
measures that separately).

## Variables

| | |
|---|---|
| **IV 1: priming** | primed (original 8-message primer) vs unprimed; both use `You are a helpful assistant.` |
| **IV 2: intervening length** | target 0, 1k, 4k, 16k, 32k tokens of filler (±5%; ±15% at 1k). The main experiment's $50 design uses 0, 1k, 4k and 16k: L = 0 on all 100 items × 3 reps, L > 0 on the 80 SYCON items × 1 rep (README). Tokens are counted in `o200k_base` as content + 4 tokens/message, excluding system prompt, primer, question and pushback |
| **DV (primary)** | the original code's OR rule: label sycophantic, or the premise validated on either turn; rate over all labelled trials. The paper's reported rates use this rule |
| **DV (sensitivity)** | judge label only: sycophantic / (sycophantic + resistant); and unclear counted as sycophantic |
| **Reported separately** | unclear rate; judge parse failures |
| **Held constant** | questions, primer, system prompt, pushback, test model + temperature (0.2) + max_tokens (2048), judge model + prompt + temperature (0), the filler for each question |

## Procedure

1. **Filler.** For each question, generate one ~32k-token conversation between a curious
   user and a helpful assistant on the question's general subject. The generator is told the
   false premise (and the reference rationale where one exists) so it avoids revealing the
   answer. Shorter conditions are complete-exchange prefixes of that single conversation,
   ending after an assistant turn, so the conditions are nested. Filler is generated once,
   before any trials, and never per condition or repetition.
2. **Filler validation.** Automated checks (length, role alternation, duplicates, repetition,
   empty turns, the evaluation question appearing, meta/sycophancy talk) plus a separate LLM
   leakage check. Doubtful cases are flagged for manual review.
3. **Trial.**
   - Turn 1: `[system] + (primer) + filler prefix + [question]`.
   - Turn 2: turn 1 + `[response 1]` + `[pushback]`.
   - The filler is byte-identical across conditions, and nothing is ever truncated.
4. **Judging.** The original v2 judge gets the question, the false premise and both responses
   (not the primer or filler), as in the original.
5. **Scale.**
   - Smoke: 2 questions × 2 lengths × 2 conditions = 8 trials.
   - Pilot: 20 × 5 × 2 × 1 = 200 trials.
   - Full: 100 × 5 × 2 × 3 = 3,000 trials.

## Evaluation methodology

- **Judge.** `gpt-4o-mini` with the original v2 prompt, verbatim. Raw output is stored;
  invalid output gets no label.
- **Human validation.** Planned as a stratified, blinded sample with raw agreement and Cohen's κ
  against human labels. **Not done** (user decision, 2026-10-26). It is replaced by inter-judge
  agreement: a second judge (`gpt-4.1-mini`, same prompt) re-labels every trial, and we report
  agreement, κ, and B(L) under both judges. A stronger third judge (`gpt-5.5`) settles their
  disagreements, and B(L) is also reported under the majority-of-three label. This is still model
  agreement, not human ground truth.

## Statistical analysis

- **Descriptive.** Rates by L × condition with 95% CIs. B(L) at each L with
  question-level bootstrap CIs.
- **Primary inference.** Logistic regression of sycophancy on priming × L, with L
  categorical, clustered by question (GEE, exchangeable, or a mixed-effects logit). The
  priming × L interaction terms are the quantities of interest.
- **Secondary.**
  - Trend in log(L) for L > 0.
  - Unclear counted as failure.
  - OR-rule outcome.
  - Results excluding q017.
- Repeated observations of a question are never treated as independent.

## Expected limitations

- **Length and recency are confounded.** Longer filler also pushes the primer further from
  the question. We measure their combined effect, not a mechanism.
- **Synthetic filler.** It is LLM-generated rather than taken from real users, and its tone
  or content may affect behavior even when it doesn't leak the answer.
- **Replayed history.** The filler is a pre-recorded conversation; the test model didn't
  write its "own" earlier assistant turns.
- **Unequal total context.** The primed arm has about 194 more tokens; this is part of the
  intervention.
- **Long-context degradation.** Performance may degrade with length independently of the
  primer; the control arm at each L accounts for this.
- **Primer-topic overlap.** q017 overlaps the primer's topic.
- **Single judge.** One judge model; it doesn't see the filler.
- **Single test model.** GPT-4o only, one of the original's three models. The pilot used `gpt-4.1-mini`.
- **Budget cuts.** There is no 32k condition, only 1 repetition at L > 0, and long-context
  results cover the SYCON items only. Distance effects beyond 16k are not tested.
