import pytest
import torch

from specdecode.interfaces import AbstractTensorDrafter, AbstractTensorVerifier
from specdecode.simulator import TensorNGramDrafter, TensorGreedyVerifier, PAD_ID


# =============================================================================
# 1. Abstract class instantiation prevention
# =============================================================================

def test_cannot_instantiate_abstract_tensor_drafter():
    """AbstractTensorDrafter must not be instantiable directly."""
    with pytest.raises(TypeError) as excinfo:
        AbstractTensorDrafter()  # type: ignore
    assert "Can't instantiate abstract class" in str(excinfo.value) or "Can't instantiate class" in str(excinfo.value)


def test_cannot_instantiate_abstract_tensor_verifier():
    """AbstractTensorVerifier must not be instantiable directly."""
    with pytest.raises(TypeError) as excinfo:
        AbstractTensorVerifier()  # type: ignore
    assert "Can't instantiate abstract class" in str(excinfo.value) or "Can't instantiate class" in str(excinfo.value)


# =============================================================================
# 2. TensorNGramDrafter — depth mode (S=1)
# =============================================================================

def test_depth_drafter_perfect_match_shape_and_values():
    """Depth mode returns a single long 2D row [1, T] with the correct continuation."""
    corpus = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    drafter = TensorNGramDrafter(corpus, n=3, num_sequences=1, draft_depth=3)

    draft = drafter.generate_draft([99, 100, 3, 4])

    assert isinstance(draft, torch.Tensor)
    assert draft.dtype == torch.long
    assert draft.shape == (1, 3)
    assert draft.tolist() == [[5, 6, 7]]


def test_depth_drafter_full_budget_single_sequence():
    """A budget of B with S=1 yields one bet of length B (shape [1, B])."""
    corpus = list(range(1, 21))  # 1..20
    drafter = TensorNGramDrafter(corpus, n=3, num_sequences=1, draft_depth=10)

    draft = drafter.generate_draft([2, 3])

    assert draft.shape == (1, 10)
    assert draft.tolist() == [[4, 5, 6, 7, 8, 9, 10, 11, 12, 13]]


def test_depth_drafter_backoff():
    """Falls back from the (n-1)-gram to a 1-gram when the longer gram has no match."""
    corpus = [1, 2, 4, 5, 6, 7]
    drafter = TensorNGramDrafter(corpus, n=3, num_sequences=1, draft_depth=2)

    draft = drafter.generate_draft([3, 4])  # [3,4] absent; backs off to [4]

    assert draft.shape == (1, 2)
    assert draft.tolist() == [[5, 6]]


def test_depth_drafter_no_match_returns_empty():
    """No match anywhere returns an empty [0, 0] tensor."""
    corpus = [1, 2, 3]
    drafter = TensorNGramDrafter(corpus, n=3, num_sequences=1, draft_depth=3)

    draft = drafter.generate_draft([99, 100])

    assert draft.shape == (0, 0)
    assert draft.numel() == 0


def test_depth_drafter_padding_near_corpus_end():
    """A continuation shorter than draft_depth is right-padded with PAD_ID."""
    corpus = [1, 2, 3, 4, 5]
    drafter = TensorNGramDrafter(corpus, n=3, num_sequences=1, draft_depth=4)

    draft = drafter.generate_draft([2, 3])  # only [4, 5] remain after [2,3]

    assert draft.shape == (1, 4)
    assert draft.tolist() == [[4, 5, PAD_ID, PAD_ID]]


# =============================================================================
# 3. TensorNGramDrafter — width mode (S>1)
# =============================================================================

def test_width_drafter_multiple_distinct_branches():
    """Width mode collects several continuations with *different* first tokens."""
    # prefix [9] is followed by 1, by 2, and by 3 at different corpus positions.
    corpus = [9, 1, 0, 9, 2, 0, 9, 3, 0]
    drafter = TensorNGramDrafter(corpus, n=2, num_sequences=3, draft_depth=2)

    draft = drafter.generate_draft([9])

    assert draft.shape == (3, 2)
    # Diverse first tokens, collected in corpus order: 9->1, 9->2, 9->3.
    first_tokens = sorted(int(row[0]) for row in draft)
    assert first_tokens == [1, 2, 3]


def test_width_drafter_dedupes_identical_continuations():
    """Identical continuations and repeated first tokens are not duplicated."""
    # [9] is followed by [1,1] twice, then by [2,2] once.
    corpus = [9, 1, 1, 0, 9, 1, 1, 0, 9, 2, 2]
    drafter = TensorNGramDrafter(corpus, n=2, num_sequences=3, draft_depth=2)

    draft = drafter.generate_draft([9])

    rows = [tuple(int(t) for t in row) for row in draft]
    assert (1, 1) in rows
    assert (2, 2) in rows
    assert len(rows) == len(set(rows))  # no duplicate rows


def test_width_drafter_fewer_candidates_than_requested():
    """When the corpus offers fewer branches than S, return only what exists."""
    corpus = [9, 1, 2, 0, 9, 1, 2]  # only one distinct first-token branch for [9]
    drafter = TensorNGramDrafter(corpus, n=2, num_sequences=4, draft_depth=2)

    draft = drafter.generate_draft([9])

    assert draft.shape[0] == 1  # fewer rows than requested S=4
    assert draft.shape[1] == 2
    assert draft.tolist() == [[1, 2]]


def test_width_drafter_pads_short_branch():
    """Within a width batch, a branch shorter than T is padded with PAD_ID."""
    # [9]->[1,1,1] (full), and a [9] near the end yields a short branch.
    corpus = [9, 1, 1, 1, 5, 9, 2]
    drafter = TensorNGramDrafter(corpus, n=2, num_sequences=2, draft_depth=3)

    draft = drafter.generate_draft([9])

    assert draft.shape == (2, 3)
    rows = [tuple(int(t) for t in row) for row in draft]
    assert (1, 1, 1) in rows
    assert (2, PAD_ID, PAD_ID) in rows


# =============================================================================
# 4. TensorGreedyVerifier
# =============================================================================

def test_tensor_verifier_all_accepted_single_row():
    """A single fully-correct row accepts all draft tokens plus a recovery token."""
    verifier = TensorGreedyVerifier()
    draft = torch.tensor([[4, 5, 6]], dtype=torch.long)
    prefix = [1, 2, 3]
    complete = [1, 2, 3, 4, 5, 6, 7]

    result = verifier.verify(draft, prefix, complete)

    assert result["accepted_count"] == 3
    assert result["rejected_count"] == 0
    assert result["accepted_tokens"] == [4, 5, 6, 7]  # + recovery 7
    assert result["chosen_sequence"] == 0


def test_tensor_verifier_partial_accept():
    """Stops at the first mismatch; rejected_count is the winner's remaining tokens."""
    verifier = TensorGreedyVerifier()
    draft = torch.tensor([[4, 5, 99, 100]], dtype=torch.long)  # 99 wrong (should be 6)
    prefix = [1, 2, 3]
    complete = [1, 2, 3, 4, 5, 6, 7, 8]

    result = verifier.verify(draft, prefix, complete)

    assert result["accepted_count"] == 2
    assert result["rejected_count"] == 2
    assert result["accepted_tokens"] == [4, 5, 6]  # + recovery 6
    assert result["chosen_sequence"] == 0


def test_tensor_verifier_full_reject_first_row():
    """When every candidate's first token is wrong, row 0 is chosen with 0 accepted."""
    verifier = TensorGreedyVerifier()
    draft = torch.tensor([[99, 100], [98, 97]], dtype=torch.long)
    prefix = [1, 2, 3]
    complete = [1, 2, 3, 4, 5, 6]

    result = verifier.verify(draft, prefix, complete)

    assert result["accepted_count"] == 0
    assert result["rejected_count"] == 2  # row 0 has 2 real tokens, none accepted
    assert result["accepted_tokens"] == [4]  # recovery only
    assert result["chosen_sequence"] == 0


def test_tensor_verifier_picks_longest_matching_candidate():
    """Among several candidates the one matching ground truth longest wins."""
    verifier = TensorGreedyVerifier()
    # Ground truth continuation after prefix [1,2,3] is 4,5,6.
    draft = torch.tensor(
        [
            [4, 99, 0],   # matches 1 (4)
            [4, 5, 6],    # matches 3 (4,5,6)  <- winner
            [4, 5, 99],   # matches 2 (4,5)
        ],
        dtype=torch.long,
    )
    prefix = [1, 2, 3]
    complete = [1, 2, 3, 4, 5, 6, 7]

    result = verifier.verify(draft, prefix, complete)

    assert result["chosen_sequence"] == 1
    assert result["accepted_count"] == 3
    assert result["accepted_tokens"] == [4, 5, 6, 7]


def test_tensor_verifier_tie_breaks_lowest_index():
    """On equal match length the lowest row index wins."""
    verifier = TensorGreedyVerifier()
    draft = torch.tensor([[4, 5, 99], [4, 5, 98]], dtype=torch.long)  # both match [4,5]
    prefix = [1, 2, 3]
    complete = [1, 2, 3, 4, 5, 6, 7]

    result = verifier.verify(draft, prefix, complete)

    assert result["chosen_sequence"] == 0
    assert result["accepted_count"] == 2


def test_tensor_verifier_ignores_pad_tokens():
    """PAD_ID terminates a row; padding is never counted as accepted or rejected."""
    verifier = TensorGreedyVerifier()
    draft = torch.tensor([[4, 5, PAD_ID]], dtype=torch.long)
    prefix = [1, 2, 3]
    complete = [1, 2, 3, 4, 5, 6]

    result = verifier.verify(draft, prefix, complete)

    assert result["accepted_count"] == 2
    assert result["rejected_count"] == 0  # only 2 real tokens, both accepted
    assert result["accepted_tokens"] == [4, 5, 6]  # + recovery 6


def test_tensor_verifier_empty_draft_returns_recovery_only():
    """An empty draft tensor behaves as 'no draft': recovery token only, chosen -1."""
    verifier = TensorGreedyVerifier()
    draft = torch.empty((0, 0), dtype=torch.long)
    prefix = [1, 2, 3]
    complete = [1, 2, 3, 4, 5]

    result = verifier.verify(draft, prefix, complete)

    assert result["accepted_count"] == 0
    assert result["rejected_count"] == 0
    assert result["accepted_tokens"] == [4]  # recovery only
    assert result["chosen_sequence"] == -1


def test_tensor_verifier_at_end_of_ground_truth():
    """No recovery token is appended when the prefix already reaches the end."""
    verifier = TensorGreedyVerifier()
    draft = torch.tensor([[4, 5]], dtype=torch.long)
    prefix = [1, 2, 3]
    complete = [1, 2, 3, 4, 5]  # nothing after the accepted draft

    result = verifier.verify(draft, prefix, complete)

    assert result["accepted_count"] == 2
    assert result["accepted_tokens"] == [4, 5]  # no recovery token


# =============================================================================
# 5. End-to-end: drafter -> verifier (depth and width)
# =============================================================================

def test_depth_then_verify_roundtrip():
    """Depth draft fed into the verifier accepts the matched run."""
    corpus = [1, 2, 3, 4, 5, 6, 7, 8]
    drafter = TensorNGramDrafter(corpus, n=3, num_sequences=1, draft_depth=3)
    verifier = TensorGreedyVerifier()

    prefix = [1, 2, 3]
    draft = drafter.generate_draft(prefix)            # -> [[4,5,6]]
    result = verifier.verify(draft, prefix, corpus)

    assert draft.shape == (1, 3)
    assert result["accepted_count"] == 3
    assert result["accepted_tokens"] == [4, 5, 6, 7]


def test_width_then_verify_selects_correct_branch():
    """Width draft hedges branches; verifier picks the one matching ground truth."""
    # After prefix [9], corpus offers branches 9->1 and 9->2; ground truth continues 9->2.
    corpus = [9, 1, 0, 9, 2, 8]
    drafter = TensorNGramDrafter(corpus, n=2, num_sequences=2, draft_depth=2)
    verifier = TensorGreedyVerifier()

    complete = [9, 2, 8]   # ground truth: after 9 comes 2 then 8
    prefix = [9]
    draft = drafter.generate_draft(prefix)
    result = verifier.verify(draft, prefix, complete)

    assert draft.shape[0] == 2
    # The [2, 8] branch should win and be fully accepted.
    assert result["accepted_count"] == 2
    assert result["accepted_tokens"][:2] == [2, 8]
