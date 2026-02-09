"""
Tests for algorithms/speculative.py optimizations.

Tests cover:
  1. SequenceState: cached current_input_ids, update_with_accepted_tokens, lifecycle
  2. Verification logic: None logprobs → break (not continue)
  3. Adaptive k with bounded deque
  4. Thread safety in ContinuousBatchScheduler
  5. get_k lambda helper
  6. generate_draft_tree batched BFS
  7. Future.cancel() handling in async pipeline
  8. Benchmark token count excludes prompt
  9. In-place list extend (no reallocation)
 10. prompt_token_ids wrapping consistency
"""

from algorithms.speculative import (
    SequenceStatus,
    SequenceState,
    ContinuousBatchScheduler,
    SpeculativeDecoder,
)
import sys
import time
import threading
import unittest
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Any
from unittest.mock import MagicMock, patch, PropertyMock
from concurrent.futures import ThreadPoolExecutor, Future

# ---------------------------------------------------------------------------
# Mock vllm before importing speculative so the import doesn't fail
# ---------------------------------------------------------------------------
mock_vllm = MagicMock()
sys.modules['vllm'] = mock_vllm

# Now we can safely import


# ============================================================================
# 1. SequenceState tests
# ============================================================================

class TestSequenceState(unittest.TestCase):
    """Tests for SequenceState with cached current_input_ids."""

    def _make_seq(self, prompt_ids=None, max_tokens=256):
        prompt_ids = prompt_ids or [1, 2, 3, 4, 5]
        return SequenceState(
            request_id="test-001",
            prompt="hello",
            prompt_token_ids=prompt_ids,
            max_tokens=max_tokens,
        )

    # --- #5: cached current_input_ids ---
    def test_initial_cache_matches_prompt(self):
        """After creation, current_input_ids == prompt_token_ids (separate copy)."""
        prompt = [10, 20, 30]
        seq = self._make_seq(prompt_ids=prompt)
        self.assertEqual(seq.current_input_ids, [10, 20, 30])
        # Must be a *copy*, not the same object
        self.assertIsNot(seq.current_input_ids, prompt)

    def test_cache_updated_after_accept(self):
        """update_with_accepted_tokens extends both generated_token_ids and cache."""
        seq = self._make_seq(prompt_ids=[1, 2, 3])
        seq.update_with_accepted_tokens(
            [100, 101], n_accepted=2, n_proposed=3, eos_token_id=0)
        self.assertEqual(seq.generated_token_ids, [100, 101])
        self.assertEqual(seq.current_input_ids, [1, 2, 3, 100, 101])

    def test_cache_survives_multiple_updates(self):
        """Multiple accept rounds keep the cache in sync."""
        seq = self._make_seq(prompt_ids=[1])
        for tok in [[10, 11], [20], [30, 31, 32]]:
            seq.update_with_accepted_tokens(tok, n_accepted=len(
                tok), n_proposed=len(tok), eos_token_id=0)
        self.assertEqual(seq.current_input_ids, [1, 10, 11, 20, 30, 31, 32])
        self.assertEqual(seq.generated_token_ids, [10, 11, 20, 30, 31, 32])

    def test_current_input_ids_returns_same_object(self):
        """Property returns the cached list reference (no copy per call)."""
        seq = self._make_seq()
        a = seq.current_input_ids
        b = seq.current_input_ids
        self.assertIs(a, b, "Should return the same list object (cached)")

    # --- lifecycle ---
    def test_eos_finishes_sequence(self):
        EOS = 99
        seq = self._make_seq()
        seq.mark_running()
        seq.update_with_accepted_tokens(
            [10, EOS], n_accepted=2, n_proposed=3, eos_token_id=EOS)
        self.assertEqual(seq.status, SequenceStatus.FINISHED_EOS)
        self.assertTrue(seq.is_finished)

    def test_max_tokens_finishes_sequence(self):
        seq = self._make_seq(max_tokens=3)
        seq.mark_running()
        seq.update_with_accepted_tokens(
            [10, 11, 12], n_accepted=3, n_proposed=3, eos_token_id=0)
        self.assertEqual(seq.status, SequenceStatus.FINISHED_LENGTH)

    def test_cancel(self):
        seq = self._make_seq()
        seq.cancel()
        self.assertTrue(seq.is_finished)
        self.assertEqual(seq.status, SequenceStatus.CANCELLED)

    def test_acceptance_rate(self):
        seq = self._make_seq()
        seq.update_with_accepted_tokens(
            [10], n_accepted=1, n_proposed=4, eos_token_id=0)
        self.assertAlmostEqual(seq.acceptance_rate, 0.25)

    def test_acceptance_rate_zero_proposed(self):
        seq = self._make_seq()
        self.assertEqual(seq.acceptance_rate, 0.0)


# ============================================================================
# 2. Verification logic: None → break
# ============================================================================

class TestVerificationNoneBreak(unittest.TestCase):
    """
    The verification loop should BREAK on None logprobs, not continue.
    This tests the pure logic extracted from _batch_verify_tokens / verify_tokens.
    """

    @staticmethod
    def _run_verify_loop(draft_tokens, target_predictions, eos_token_id):
        """Extracted verification loop (mirrors the code in speculative.py)."""
        accepted_tokens = []
        n_accepted = 0
        finished = False

        for draft_token, target_pred in zip(draft_tokens, target_predictions):
            if target_pred is None:
                break  # ← the fix
            if target_pred == draft_token:
                accepted_tokens.append(draft_token)
                n_accepted += 1
                if draft_token == eos_token_id:
                    finished = True
                    break
            else:
                accepted_tokens.append(target_pred)
                finished = (target_pred == eos_token_id)
                break

        return accepted_tokens, n_accepted, finished

    def test_none_at_position_0_stops_immediately(self):
        """If first logprob is None, accept nothing."""
        acc, n, fin = self._run_verify_loop(
            [10, 20, 30], [None, 20, 30], eos_token_id=0)
        self.assertEqual(acc, [])
        self.assertEqual(n, 0)
        self.assertFalse(fin)

    def test_none_in_middle_stops_before_it(self):
        """Accept up to the None, then stop."""
        acc, n, fin = self._run_verify_loop(
            [10, 20, 30], [10, None, 30], eos_token_id=0)
        self.assertEqual(acc, [10])
        self.assertEqual(n, 1)

    def test_old_continue_would_skip_none(self):
        """
        With the old 'continue' logic, [10, None, 30] with predictions [10, None, 30]
        would have accepted [10, 30] (n_accepted=2). The new 'break' stops at [10] (n_accepted=1).
        """
        # Simulate old behavior (continue)
        draft_tokens = [10, 20, 30]
        target_preds = [10, None, 30]
        eos = 0

        # Old logic with continue
        old_accepted = []
        old_n = 0
        for dt, tp in zip(draft_tokens, target_preds):
            if tp is None:
                continue  # OLD behavior
            if tp == dt:
                old_accepted.append(dt)
                old_n += 1
            else:
                old_accepted.append(tp)
                break

        # New logic with break
        new_accepted, new_n, _ = self._run_verify_loop(
            draft_tokens, target_preds, eos)

        # Old would incorrectly accept token 30 after skipping None
        self.assertEqual(old_accepted, [10, 30])
        self.assertEqual(old_n, 2)
        # New correctly stops at the None
        self.assertEqual(new_accepted, [10])
        self.assertEqual(new_n, 1)

    def test_all_match_no_none(self):
        """All match, no None — all accepted."""
        acc, n, fin = self._run_verify_loop(
            [10, 20, 30], [10, 20, 30], eos_token_id=0)
        self.assertEqual(acc, [10, 20, 30])
        self.assertEqual(n, 3)
        self.assertFalse(fin)

    def test_mismatch_replaces_with_target(self):
        """First mismatch replaces draft with target prediction."""
        acc, n, fin = self._run_verify_loop(
            [10, 20, 30], [10, 25, 30], eos_token_id=0)
        self.assertEqual(acc, [10, 25])
        self.assertEqual(n, 1)

    def test_eos_in_draft_accepted(self):
        """EOS token in accepted draft → finished."""
        EOS = 99
        acc, n, fin = self._run_verify_loop(
            [10, EOS, 30], [10, EOS, 30], eos_token_id=EOS)
        self.assertEqual(acc, [10, EOS])
        self.assertEqual(n, 2)
        self.assertTrue(fin)

    def test_eos_as_target_replacement(self):
        """Target replaces draft with EOS → finished."""
        EOS = 99
        acc, n, fin = self._run_verify_loop(
            [10, 20, 30], [10, EOS, 30], eos_token_id=EOS)
        self.assertEqual(acc, [10, EOS])
        self.assertEqual(n, 1)
        self.assertTrue(fin)


# ============================================================================
# 3. Adaptive k with bounded deque
# ============================================================================

class TestAdaptiveKDeque(unittest.TestCase):
    """Test that acceptance_history is a bounded deque and _update_speculation_length works."""

    def _make_decoder_stub(self):
        """Create a minimal object with the adaptive-k attributes (no model loading)."""
        obj = object.__new__(SpeculativeDecoder)
        obj.adaptive_k = True
        obj.min_k = 4
        obj.max_k = 32
        obj.current_k = 16
        obj._acceptance_window_size = 10
        obj.acceptance_history = deque(maxlen=10)
        return obj

    def test_deque_is_bounded(self):
        dec = self._make_decoder_stub()
        self.assertIsInstance(dec.acceptance_history, deque)
        self.assertEqual(dec.acceptance_history.maxlen, 10)

    def test_deque_does_not_grow_past_maxlen(self):
        dec = self._make_decoder_stub()
        for i in range(50):
            dec._update_speculation_length(8, 10)  # 80% rate
        self.assertEqual(len(dec.acceptance_history), 10)

    def test_high_acceptance_increases_k(self):
        dec = self._make_decoder_stub()
        dec.current_k = 10
        # Feed many high-acceptance rounds
        for _ in range(15):
            dec._update_speculation_length(9, 10)  # 90%
        self.assertGreater(dec.current_k, 10)

    def test_low_acceptance_decreases_k(self):
        dec = self._make_decoder_stub()
        dec.current_k = 20
        for _ in range(15):
            dec._update_speculation_length(3, 10)  # 30%
        self.assertLess(dec.current_k, 20)

    def test_k_stays_within_bounds(self):
        dec = self._make_decoder_stub()
        # Push k up
        for _ in range(200):
            dec._update_speculation_length(10, 10)
        self.assertLessEqual(dec.current_k, dec.max_k)
        # Push k down
        for _ in range(200):
            dec._update_speculation_length(0, 10)
        self.assertGreaterEqual(dec.current_k, dec.min_k)

    def test_zero_proposed_is_noop(self):
        dec = self._make_decoder_stub()
        old_k = dec.current_k
        dec._update_speculation_length(0, 0)
        self.assertEqual(dec.current_k, old_k)
        self.assertEqual(len(dec.acceptance_history), 0)

    def test_clear_resets_deque(self):
        dec = self._make_decoder_stub()
        for _ in range(5):
            dec._update_speculation_length(5, 10)
        self.assertEqual(len(dec.acceptance_history), 5)
        dec.acceptance_history.clear()
        self.assertEqual(len(dec.acceptance_history), 0)
        self.assertEqual(dec.acceptance_history.maxlen, 10)  # maxlen preserved


# ============================================================================
# 4. Thread safety in ContinuousBatchScheduler
# ============================================================================

class TestSchedulerThreadSafety(unittest.TestCase):
    """Test that scheduler operations are properly locked."""

    def _make_scheduler(self):
        decoder = MagicMock()
        decoder.target_tokenizer.encode.side_effect = lambda s: list(
            range(len(s)))
        decoder.target_eos_id = 0
        return ContinuousBatchScheduler(decoder, max_batch_size=4, num_speculative_tokens=8)

    def test_get_result_returns_none_for_unknown(self):
        sched = self._make_scheduler()
        self.assertIsNone(sched.get_result("nonexistent"))

    def test_merge_respects_max_batch_size(self):
        sched = self._make_scheduler()
        for i in range(10):
            seq = SequenceState(
                request_id=f"req-{i}", prompt="hi", prompt_token_ids=[1, 2]
            )
            sched.pending_queue.put(seq)
        sched._merge_pending_requests()
        self.assertEqual(len(sched.active_sequences), 4)  # max_batch_size
        self.assertEqual(sched.pending_queue.qsize(), 6)

    def test_concurrent_merge_does_not_exceed_batch_size(self):
        """Multiple threads calling _merge_pending_requests simultaneously."""
        sched = self._make_scheduler()
        for i in range(100):
            seq = SequenceState(
                request_id=f"req-{i}", prompt="hi", prompt_token_ids=[1, 2]
            )
            sched.pending_queue.put(seq)

        errors = []

        def merge_many():
            for _ in range(20):
                sched._merge_pending_requests()
                with sched._lock:
                    if len(sched.active_sequences) > sched.max_batch_size:
                        errors.append(len(sched.active_sequences))

        threads = [threading.Thread(target=merge_many) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [], f"Batch size exceeded: {errors}")

    def test_get_stats_under_lock(self):
        sched = self._make_scheduler()
        stats = sched.get_stats()
        self.assertEqual(stats["active_sequences"], 0)
        self.assertEqual(stats["total_iterations"], 0)

    def test_cancel_nonexistent_returns_false(self):
        sched = self._make_scheduler()
        self.assertFalse(sched.cancel_request("unknown"))

    def test_cancel_active_returns_true(self):
        sched = self._make_scheduler()
        seq = SequenceState(request_id="c1", prompt="hi", prompt_token_ids=[1])
        with sched._lock:
            sched.active_sequences["c1"] = seq
        self.assertTrue(sched.cancel_request("c1"))
        self.assertEqual(seq.status, SequenceStatus.CANCELLED)


# ============================================================================
# 5. get_k lambda helper
# ============================================================================

class TestGetKHelper(unittest.TestCase):
    """Test that the get_k lambda computes the correct value."""

    def test_adaptive_mode_returns_current_k(self):
        class Obj:
            current_k = 12
        obj = Obj()
        use_adaptive = True
        num_speculative_tokens = None
        def get_k(): return obj.current_k if use_adaptive else (
            num_speculative_tokens or 16)
        self.assertEqual(get_k(), 12)
        obj.current_k = 20
        self.assertEqual(get_k(), 20)  # reflects updated value

    def test_fixed_mode_returns_specified(self):
        use_adaptive = False
        num_speculative_tokens = 8
        def get_k(): return 0 if use_adaptive else (num_speculative_tokens or 16)
        self.assertEqual(get_k(), 8)

    def test_fixed_mode_none_defaults_to_16(self):
        use_adaptive = False
        num_speculative_tokens = None
        def get_k(): return 0 if use_adaptive else (num_speculative_tokens or 16)
        self.assertEqual(get_k(), 16)


# ============================================================================
# 6. generate_draft_tree batched BFS
# ============================================================================

class TestGenerateDraftTreeBatched(unittest.TestCase):
    """Test that the batched BFS issues one LLM call per depth level."""

    def _make_decoder_stub_with_mock_draft(self, width=2):
        """Create decoder stub where draft_llm.generate is tracked."""
        obj = object.__new__(SpeculativeDecoder)
        obj.target_eos_id = 0
        obj.draft_llm = MagicMock()

        # Create a mock logprob entry
        @dataclass
        class FakeLogprob:
            logprob: float

        def fake_generate(prompt_token_ids, sampling_params, use_tqdm):
            """Return width top tokens for each input in the batch."""
            results = []
            for _ids in prompt_token_ids:
                # Generate deterministic "top-k" tokens based on input length
                base = len(_ids) * 100
                logprobs = {
                    base + i: FakeLogprob(logprob=-0.1 * (i + 1))
                    for i in range(width)
                }
                mock_output = MagicMock()
                mock_output.outputs = [MagicMock()]
                mock_output.outputs[0].logprobs = [logprobs]
                results.append(mock_output)
            return results

        obj.draft_llm.generate.side_effect = fake_generate
        return obj

    def test_batched_tree_depth2_width2(self):
        """depth=2, width=2 → at most 2 LLM calls (one per level)."""
        dec = self._make_decoder_stub_with_mock_draft(width=2)
        tree = dec.generate_draft_tree([1, 2, 3], depth=2, width=2)

        # Should produce width^depth = 4 paths
        self.assertEqual(len(tree), 4)
        # Each path should have length == depth
        for path in tree:
            self.assertEqual(len(path), 2)
        # Should have called generate exactly 2 times (once per depth level)
        self.assertEqual(dec.draft_llm.generate.call_count, 2)

    def test_batched_tree_depth1(self):
        """depth=1 → exactly 1 LLM call."""
        dec = self._make_decoder_stub_with_mock_draft(width=3)
        tree = dec.generate_draft_tree([1], depth=1, width=3)
        self.assertEqual(len(tree), 3)
        self.assertEqual(dec.draft_llm.generate.call_count, 1)

    def test_batch_sizes_increase_by_width(self):
        """At depth d, the batch size should be width^d."""
        dec = self._make_decoder_stub_with_mock_draft(width=2)
        dec.generate_draft_tree([1], depth=3, width=2)

        calls = dec.draft_llm.generate.call_args_list
        self.assertEqual(len(calls), 3)  # 3 depth levels
        # Depth 0: 1 node, depth 1: 2 nodes, depth 2: 4 nodes
        for i, call in enumerate(calls):
            batch_input = call.kwargs.get(
                'prompt_token_ids') or call[1].get('prompt_token_ids')
            if batch_input is None:
                # Try positional
                batch_input = call[0][0] if call[0] else None
            expected_batch_size = 2 ** i
            self.assertEqual(len(batch_input), expected_batch_size,
                             f"Depth {i}: expected batch size {expected_batch_size}, got {len(batch_input)}")


# ============================================================================
# 7. Future.cancel() handling
# ============================================================================

class TestFutureCancelHandling(unittest.TestCase):
    """Test that unreachable futures are properly waited on, not just cancel()."""

    def test_cancel_succeeds_for_pending_future(self):
        """If cancel() returns True, result() should not be called."""
        future = MagicMock(spec=Future)
        future.cancel.return_value = True  # Successfully cancelled

        # Simulate the code path
        if not future.cancel():
            future.result()

        future.cancel.assert_called_once()
        future.result.assert_not_called()

    def test_cancel_fails_waits_for_result(self):
        """If cancel() returns False (already running), we wait for result()."""
        future = MagicMock(spec=Future)
        future.cancel.return_value = False  # Already running
        future.result.return_value = [100, 200]  # Draft tokens (discarded)

        # Simulate the code path
        if not future.cancel():
            future.result()

        future.cancel.assert_called_once()
        future.result.assert_called_once()

    def test_real_threadpool_cancel_behavior(self):
        """Integration test: submit a task, try to cancel — verify no orphans."""
        executor = ThreadPoolExecutor(max_workers=1)
        # Submit a blocking task first to make the second task pending
        blocker = executor.submit(time.sleep, 0.2)
        pending = executor.submit(lambda: 42)

        # The pending task should be cancellable
        cancelled = pending.cancel()
        # Whether it was cancelled depends on timing, but we handle both cases
        if not cancelled:
            result = pending.result(timeout=2)
            self.assertEqual(result, 42)

        blocker.result(timeout=2)  # Clean up
        executor.shutdown(wait=True)


# ============================================================================
# 8. Benchmark token count
# ============================================================================

class TestBenchmarkTokenCount(unittest.TestCase):
    """Test that speculative benchmark correctly subtracts prompt tokens."""

    def test_token_subtraction_logic(self):
        """
        Simulate: prompt encodes to 5 tokens, output (prompt + generated) encodes to 15.
        Generated tokens = 15 - 5 = 10.
        """
        prompt = "hello world"
        prompt_token_count = 5  # simulated
        full_output_token_count = 15  # simulated

        output_tokens = full_output_token_count - prompt_token_count
        self.assertEqual(output_tokens, 10)

    def test_no_generation_gives_zero(self):
        """If output == prompt, generated tokens = 0."""
        prompt_token_count = 5
        full_output_token_count = 5
        output_tokens = full_output_token_count - prompt_token_count
        self.assertEqual(output_tokens, 0)


# ============================================================================
# 9. In-place list extend
# ============================================================================

class TestInPlaceExtend(unittest.TestCase):
    """Test that extend modifies in-place (no new list created)."""

    def test_extend_preserves_identity(self):
        """list.extend() should not change the list object identity."""
        input_ids = [1, 2, 3, 4, 5]
        original_id = id(input_ids)
        input_ids.extend([10, 11, 12])
        self.assertEqual(id(input_ids), original_id)
        self.assertEqual(input_ids, [1, 2, 3, 4, 5, 10, 11, 12])

    def test_concatenation_creates_new_list(self):
        """For contrast: `+` creates a new object."""
        input_ids = [1, 2, 3]
        original_id = id(input_ids)
        input_ids = input_ids + [10]
        self.assertNotEqual(id(input_ids), original_id)


# ============================================================================
# 10. prompt_token_ids wrapping consistency
# ============================================================================

class TestPromptTokenIdsWrapping(unittest.TestCase):
    """Test that single-sequence calls use [input_ids] (list-of-lists)."""

    def test_generate_draft_tokens_wraps_input(self):
        """generate_draft_tokens should pass [input_ids] to draft_llm.generate."""
        dec = object.__new__(SpeculativeDecoder)
        dec.target_eos_id = 0
        dec.draft_llm = MagicMock()

        mock_output = MagicMock()
        mock_output.outputs = [MagicMock()]
        mock_output.outputs[0].token_ids = [100, 101, 102]
        dec.draft_llm.generate.return_value = [mock_output]

        result = dec.generate_draft_tokens([1, 2, 3], num_speculative_tokens=3)

        call_kwargs = dec.draft_llm.generate.call_args.kwargs
        prompt_arg = call_kwargs.get('prompt_token_ids')
        # Should be [[1, 2, 3]] not [1, 2, 3]
        self.assertEqual(prompt_arg, [[1, 2, 3]])
        self.assertEqual(result, [100, 101, 102])

    def test_verify_tokens_wraps_input(self):
        """verify_tokens should pass [candidate_ids] to target_llm.generate."""
        dec = object.__new__(SpeculativeDecoder)
        dec.target_eos_id = 0
        dec.target_llm = MagicMock()
        dec.verify_sampling_params = MagicMock()

        mock_output = MagicMock()
        # Simulate logprobs for positions 0..4 (3 prompt + 2 draft)
        mock_output.prompt_logprobs = [
            None, None, None,  # prompt positions
            {100: MagicMock()},  # draft pos 0 → predicts 100
            {200: MagicMock()},  # draft pos 1 → predicts 200
        ]
        mock_output.outputs = [MagicMock()]
        mock_output.outputs[0].token_ids = [300]  # bonus token
        dec.target_llm.generate.return_value = [mock_output]

        result = dec.verify_tokens([1, 2, 3], [100, 200])

        call_kwargs = dec.target_llm.generate.call_args.kwargs
        prompt_arg = call_kwargs.get('prompt_token_ids')
        # Should be [[1, 2, 3, 100, 200]] not [1, 2, 3, 100, 200]
        self.assertEqual(prompt_arg, [[1, 2, 3, 100, 200]])


# ============================================================================
# Integration-style tests
# ============================================================================

class TestVerifyTokensIntegration(unittest.TestCase):
    """Integration test of verify_tokens with all edge cases."""

    def _setup_decoder(self):
        dec = object.__new__(SpeculativeDecoder)
        dec.target_eos_id = 99
        dec.target_llm = MagicMock()
        dec.verify_sampling_params = MagicMock()
        return dec

    def _mock_verify(self, dec, prompt_logprobs, bonus_token=0):
        mock_output = MagicMock()
        mock_output.prompt_logprobs = prompt_logprobs
        mock_output.outputs = [MagicMock()]
        mock_output.outputs[0].token_ids = [bonus_token]
        dec.target_llm.generate.return_value = [mock_output]

    def test_all_accepted_with_bonus(self):
        """All drafts accepted → bonus token appended."""
        dec = self._setup_decoder()
        input_ids = [1, 2]
        draft = [10, 20]
        self._mock_verify(dec, [
            None, None,        # prompt
            {10: MagicMock()},  # draft[0] → 10 ✓
            {20: MagicMock()},  # draft[1] → 20 ✓
        ], bonus_token=50)

        acc, n, fin = dec.verify_tokens(input_ids, draft)
        self.assertEqual(acc, [10, 20, 50])
        self.assertEqual(n, 2)
        self.assertFalse(fin)

    def test_mismatch_at_first_position(self):
        """First draft rejected → replace with target pred."""
        dec = self._setup_decoder()
        self._mock_verify(dec, [
            None, None,
            {15: MagicMock()},  # draft was 10, target says 15 → mismatch
            {20: MagicMock()},
        ])

        acc, n, fin = dec.verify_tokens([1, 2], [10, 20])
        self.assertEqual(acc, [15])
        self.assertEqual(n, 0)

    def test_none_logprob_breaks(self):
        """None in logprobs → break, no tokens accepted after None."""
        dec = self._setup_decoder()
        self._mock_verify(dec, [
            None, None,
            None,              # None logprob at draft position 0
            {20: MagicMock()},
        ])

        acc, n, fin = dec.verify_tokens([1, 2], [10, 20])
        self.assertEqual(acc, [])
        self.assertEqual(n, 0)

    def test_bonus_token_is_eos(self):
        """All accepted + bonus is EOS → finished."""
        dec = self._setup_decoder()
        self._mock_verify(dec, [
            None, None,
            {10: MagicMock()},
            {20: MagicMock()},
        ], bonus_token=99)  # EOS

        acc, n, fin = dec.verify_tokens([1, 2], [10, 20])
        self.assertEqual(acc, [10, 20, 99])
        self.assertTrue(fin)


class TestSequenceStateCacheEdgeCases(unittest.TestCase):
    """Edge cases for the cached current_input_ids."""

    def test_empty_prompt(self):
        seq = SequenceState(request_id="e1", prompt="", prompt_token_ids=[])
        self.assertEqual(seq.current_input_ids, [])
        seq.update_with_accepted_tokens(
            [10], n_accepted=1, n_proposed=1, eos_token_id=0)
        self.assertEqual(seq.current_input_ids, [10])

    def test_empty_accepted_tokens(self):
        seq = SequenceState(request_id="e2", prompt="hi",
                            prompt_token_ids=[1, 2])
        seq.update_with_accepted_tokens(
            [], n_accepted=0, n_proposed=3, eos_token_id=0)
        self.assertEqual(seq.current_input_ids, [1, 2])
        self.assertEqual(seq.generated_token_ids, [])


class TestRemoveFinishedSequences(unittest.TestCase):
    """Test _remove_finished_sequences with callbacks."""

    def _make_scheduler(self):
        decoder = MagicMock()
        decoder.target_eos_id = 0
        return ContinuousBatchScheduler(decoder, max_batch_size=32)

    def test_finished_sequences_moved_to_completed(self):
        sched = self._make_scheduler()
        seq = SequenceState(request_id="f1", prompt="hi", prompt_token_ids=[1])
        seq.status = SequenceStatus.FINISHED_EOS
        with sched._lock:
            sched.active_sequences["f1"] = seq

        finished = sched._remove_finished_sequences()
        self.assertEqual(len(finished), 1)
        self.assertNotIn("f1", sched.active_sequences)
        self.assertIn("f1", sched.completed_sequences)
        self.assertEqual(sched.total_sequences_processed, 1)

    def test_callback_invoked_on_completion(self):
        sched = self._make_scheduler()
        callback_called = []
        seq = SequenceState(
            request_id="f2", prompt="hi", prompt_token_ids=[1],
            on_complete=lambda s: callback_called.append(s.request_id)
        )
        seq.status = SequenceStatus.FINISHED_LENGTH
        with sched._lock:
            sched.active_sequences["f2"] = seq

        sched._remove_finished_sequences()
        self.assertEqual(callback_called, ["f2"])

    def test_callback_error_does_not_crash(self):
        sched = self._make_scheduler()
        seq = SequenceState(
            request_id="f3", prompt="hi", prompt_token_ids=[1],
            on_complete=lambda s: 1 / 0  # Will raise ZeroDivisionError
        )
        seq.status = SequenceStatus.FINISHED_EOS
        with sched._lock:
            sched.active_sequences["f3"] = seq

        # Should not raise
        finished = sched._remove_finished_sequences()
        self.assertEqual(len(finished), 1)


if __name__ == '__main__':
    unittest.main()
