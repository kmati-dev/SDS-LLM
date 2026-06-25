"""
Hardcoded wiki-style demo dataset — no download required.
Corpus: technical article about speculative decoding.
Target: shorter rephrased version of the same topic.
"""


def load(index: int = 0) -> tuple:
    corpus = """
    Speculative decoding is a powerful technique designed to accelerate Large Language Model (LLM) inference.
    Standard LLM decoding generates tokens one by one in an autoregressive fashion, which is heavily memory-bound.
    Each forward pass of a large model is computationally expensive and slow because weights must be loaded from
    high-bandwidth memory to the GPU SRAM for every single token generated.

    To solve this bottleneck, speculative decoding introduces a dual-model framework consisting of a small, fast
    draft model (the Drafter) and a large, high-capacity target model (the Verifier). The fast draft model guesses
    a sequence of K future tokens (speculative tokens) at a very low cost.
    Then, the large target model runs a single parallel forward pass to verify all K speculative tokens in parallel.
    Since the target model verifies K tokens in a single step, if the drafts are accepted, we get K tokens for the
    computational cost of just one target model step, leading to substantial speedups.

    In greedy speculative decoding, we compare the speculative tokens directly against the argmax predictions of the
    target model. We accept tokens sequentially until the first mismatch occurs. If a mismatch is found at index i,
    we reject all subsequent tokens, accept the matched tokens, and append a recovery token provided by the target model's
    ground truth. This ensures that the final output matches the exact distribution of the target model perfectly
    while saving computation steps.
    """

    target = """
    Speculative decoding is a powerful technique designed to accelerate Large Language Model inference.
    Each forward pass of a large model is computationally expensive.
    To solve this bottleneck, speculative decoding introduces a draft model and a large target model.
    The draft model guesses a sequence of speculative tokens at a low cost.
    Then, the large target model runs a single parallel forward pass to verify speculative tokens.
    In greedy speculative decoding, we accept tokens sequentially until the first mismatch.
    This ensures that the final output matches the exact distribution.
    """

    return corpus, target
