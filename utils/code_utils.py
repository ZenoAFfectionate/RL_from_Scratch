import ast
import io
import json
import os
import re
import signal
import subprocess
import sys
import textwrap
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple, Union


# Pre-compiled regex patterns for code extraction.
_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_TAG_PATTERNS = [
    re.compile(r"<code>(.*?)</code>",        re.DOTALL),
    re.compile(r"<answer>(.*?)</answer>",     re.DOTALL),
    re.compile(r"<solution>(.*?)</solution>", re.DOTALL),
]
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def extract_code_from_response(response: str) -> Optional[str]:
    """Extract Python code from a model response.

    Handles fenced code blocks, <code>/<answer>/<solution> tags, and raw code.
    Returns ``None`` only when the response is empty / whitespace.
    """
    if not response or not response.strip():
        return None

    matches = _FENCE_RE.findall(response)
    if matches:
        return matches[-1].strip()

    for pat in _TAG_PATTERNS:
        matches = pat.findall(response)
        if matches:
            code = matches[-1].strip()
            inner = extract_code_from_response(code)
            return inner if inner is not None else code

    cleaned = _THINK_RE.sub("", response).strip()
    # Strip orphan closing tags left over when the opening tag was in the prompt
    cleaned = re.sub(r"</solution>\s*$", "", cleaned).strip()
    return cleaned or response.strip() or None


def check_syntax(code: str) -> Tuple[bool, Optional[str]]:
    """Check whether *code* is syntactically valid Python."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            ast.parse(code)
        return True, None
    except SyntaxError as e:
        return False, f"SyntaxError: {e.msg} (line {e.lineno})"
    except Exception as e:
        return False, str(e)


# ---------------------------------------------------------------------------
# Subprocess helpers — all code execution uses isolated subprocesses with
# process-group killing to prevent deadlocks and resource leaks.
#
# NOTE: We intentionally avoid persistent process pools (multiprocessing.Pool)
# because LLM-generated code can hang workers in ways that SIGALRM cannot
# interrupt (C-extension loops, blocking I/O, signal overrides).  Stuck
# workers accumulate over thousands of evaluations and eventually deadlock
# the pool.  Fresh subprocesses with start_new_session=True + os.killpg()
# guarantee that every evaluation is fully cleaned up on timeout.
# ---------------------------------------------------------------------------

def _fail(error: str, timeout: bool = False) -> Dict[str, Any]:
    """Shorthand for a failed-test result dict."""
    return {"passed": False, "error": error, "stdout": "", "timeout": timeout}


def _make_summary(details: List[Dict[str, Any]], total: int) -> Dict[str, Any]:
    """Build the standard summary dict from per-test results."""
    passed_count = sum(1 for r in details if r.get("passed"))
    return {
        "passed": passed_count == total,
        "total": total,
        "passed_count": passed_count,
        "failed_count": total - passed_count,
        "error": None if passed_count == total else next(
            (r.get("error") for r in details if not r.get("passed")), None),
        "details": details,
    }


def _safe_subprocess_run(
    args: List[str],
    input_data: Optional[str] = None,
    timeout: float = 10.0,
) -> Tuple[str, str, int, bool]:
    """Run a subprocess in a new session/process-group.

    On timeout, kills the **entire process group** (including any child
    processes the executed code may have spawned), preventing zombie
    processes and resource leaks.

    Returns ``(stdout, stderr, returncode, timed_out)``.
    """
    proc = subprocess.Popen(
        args,
        stdin=subprocess.PIPE if input_data is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,  # new process group for reliable killing
    )
    try:
        stdout, stderr = proc.communicate(input=input_data, timeout=timeout)
        return stdout, stderr, proc.returncode, False
    except subprocess.TimeoutExpired:
        _kill_proc_group(proc)
        return "", "", -1, True
    except Exception:
        _kill_proc_group(proc)
        raise


def _kill_proc_group(proc: subprocess.Popen) -> None:
    """Kill an entire process group and wait for cleanup."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.kill()
        except (ProcessLookupError, OSError):
            pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass


# ---------------------------------------------------------------------------
# Assert-based tests (MBPP / KodCode style)
# ---------------------------------------------------------------------------

_BATCH_ASSERT_TEMPLATE = textwrap.dedent(r'''
import sys, io, json, resource, signal, warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)

MAX_MEM = {max_memory}
MAX_CPU = {cpu_timeout}
try:
    resource.setrlimit(resource.RLIMIT_AS, (MAX_MEM, MAX_MEM))
except Exception:
    pass
try:
    resource.setrlimit(resource.RLIMIT_CPU, (MAX_CPU, MAX_CPU))
except Exception:
    pass

_code = {code_repr}
_tests = {tests_repr}

_globals = {{"__builtins__": __builtins__}}
_real_stdout = sys.stdout
sys.stdout = io.StringIO()
try:
    exec(compile(_code, "<candidate>", "exec"), _globals)
except Exception as _e:
    sys.stdout = _real_stdout
    _err = {{"passed": False, "error": str(_e), "stdout": "", "timeout": False}}
    print(json.dumps([_err] * len(_tests)))
    sys.exit(0)
finally:
    sys.stdout = _real_stdout

_results = []
for _t in _tests:
    _cap = io.StringIO()
    _old = sys.stdout
    sys.stdout = _cap
    try:
        exec(compile(_t, "<test>", "exec"), _globals)
        sys.stdout = _old
        _results.append({{"passed": True, "error": None, "stdout": _cap.getvalue(), "timeout": False}})
    except Exception as _e:
        sys.stdout = _old
        _results.append({{"passed": False, "error": str(_e), "stdout": _cap.getvalue(), "timeout": False}})

print(json.dumps(_results))
sys.exit(0)
''')


def _run_batch_asserts_subprocess(
    code: str, tests: List[str],
    timeout: float = 10.0,
    max_memory: int = 512 * 1024 * 1024,
    cpu_timeout: int = 10,
) -> List[Dict[str, Any]]:
    """Run all asserts in a single isolated subprocess."""
    script = _BATCH_ASSERT_TEMPLATE.format(
        code_repr=repr(code),
        tests_repr=repr(tests),
        max_memory=max_memory,
        cpu_timeout=cpu_timeout,
    )
    stdout, stderr, rc, timed_out = _safe_subprocess_run(
        [sys.executable, "-S", "-W", "ignore::SyntaxWarning", "-c", script],
        timeout=timeout,
    )
    if timed_out:
        return [_fail("Execution timed out", timeout=True) for _ in tests]
    try:
        return json.loads(stdout.strip())
    except (json.JSONDecodeError, ValueError) as e:
        return [_fail(f"Output parse error: {e}") for _ in tests]


def run_assert_tests(code: str, test_list: Union[str, List[str]],
                     timeout: float = 10.0) -> Dict[str, Any]:
    """Run assertion-based tests (MBPP / KodCode style) in an isolated subprocess."""
    if isinstance(test_list, str):
        test_list = [test_list]
    tests = [t.strip() for t in test_list]
    details = _run_batch_asserts_subprocess(code, tests, timeout=timeout)
    return _make_summary(details, len(tests))


# ---------------------------------------------------------------------------
# Stdin/stdout-based tests (APPS / CodeContests style)
# ---------------------------------------------------------------------------

def _run_stdin_stdout_single(
    code: str, stdin_data: str, expected_stdout: str, timeout: float
) -> Dict[str, Any]:
    """Execute *code* with *stdin_data* and compare stdout to *expected_stdout*."""
    stdout, stderr, rc, timed_out = _safe_subprocess_run(
        [sys.executable, "-S", "-W", "ignore::SyntaxWarning", "-c", code],
        input_data=stdin_data,
        timeout=timeout,
    )
    if timed_out:
        return {
            "passed": False,
            "actual_output": "",
            "expected_output": expected_stdout.strip(),
            "error": "Execution timed out",
            "timeout": True,
        }
    actual = stdout.strip()
    expected = expected_stdout.strip()
    return {
        "passed": actual == expected,
        "actual_output": actual,
        "expected_output": expected,
        "error": stderr.strip() if rc != 0 else None,
        "timeout": False,
    }


def run_stdin_stdout_tests(
    code: str,
    test_cases: Union[str, Dict, List[Dict]],
    timeout: float = 10.0,
) -> Dict[str, Any]:
    """
    Run stdin/stdout-based tests (APPS / CodeContests style).

    Parameters:
        code:       The candidate solution code.
        test_cases: Either a JSON string or a dict/list with keys
                    ``"inputs"`` and ``"outputs"`` (each a list of strings).
        timeout:    Wall-clock timeout in seconds.

    Returns dict with:
        passed (bool), total (int), passed_count (int),
        failed_count (int), error (str|None), details (list)
    """
    # Parse test_cases into a standard form.
    if isinstance(test_cases, str):
        try:
            test_cases = json.loads(test_cases)
        except json.JSONDecodeError:
            return {
                "passed": False, "total": 0, "passed_count": 0,
                "failed_count": 0, "error": "Invalid test_cases JSON",
                "details": [],
            }

    # Normalise: support both {"inputs": [...], "outputs": [...]}
    # and [{"input": ..., "output": ...}, ...].
    if isinstance(test_cases, dict):
        inputs  = test_cases.get("inputs",  test_cases.get("input",  []))
        outputs = test_cases.get("outputs", test_cases.get("output", []))
        if isinstance(inputs, str):
            inputs = [inputs]
        if isinstance(outputs, str):
            outputs = [outputs]
    elif isinstance(test_cases, list):
        inputs  = [tc.get("input",  tc.get("inputs",  "")) for tc in test_cases]
        outputs = [tc.get("output", tc.get("outputs", "")) for tc in test_cases]
    else:
        return {
            "passed": False, "total": 0, "passed_count": 0,
            "failed_count": 0, "error": "Unrecognised test_cases format",
            "details": [],
        }

    # Normalize all inputs/outputs to strings first.
    norm_inputs = []
    norm_outputs = []
    for inp, expected_out in zip(inputs, outputs):
        if isinstance(inp, list):
            inp = "\n".join(str(x) for x in inp) + "\n"
        if isinstance(expected_out, list):
            expected_out = "\n".join(str(x) for x in expected_out) + "\n"
        norm_inputs.append(inp)
        norm_outputs.append(expected_out)

    # Parallel execution using threads.  Each thread spawns an isolated
    # subprocess (via _safe_subprocess_run), so threads provide real
    # parallelism because the GIL is released during subprocess I/O.
    if len(norm_inputs) > 1:
        max_workers = min(os.cpu_count() or 4, 8)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    _run_stdin_stdout_single, code, inp, exp, timeout
                ): i
                for i, (inp, exp) in enumerate(zip(norm_inputs, norm_outputs))
            }
            indexed_results: Dict[int, Dict[str, Any]] = {}
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    indexed_results[idx] = future.result()
                except Exception as e:
                    indexed_results[idx] = {
                        "passed": False, "actual_output": "",
                        "expected_output": norm_outputs[idx].strip(),
                        "error": str(e), "timeout": False,
                    }
            details = [indexed_results[i] for i in range(len(norm_inputs))]
    else:
        details = [
            _run_stdin_stdout_single(code, inp, exp, timeout)
            for inp, exp in zip(norm_inputs, norm_outputs)
        ]

    total = len(norm_inputs)
    return _make_summary(details, total)


# ---------------------------------------------------------------------------
# Unified test runner & end-to-end evaluator
# ---------------------------------------------------------------------------

def run_tests(
    code: str,
    test_cases: Any,
    timeout: float = 10.0,
    test_type: Optional[str] = None,
) -> Dict[str, Any]:
    """Unified test runner that auto-detects the test format."""
    if test_type == "assert":
        return run_assert_tests(code, test_cases, timeout)
    if test_type == "stdin_stdout":
        return run_stdin_stdout_tests(code, test_cases, timeout)

    # String -> try JSON parse first.
    if isinstance(test_cases, str):
        stripped = test_cases.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            try:
                parsed = json.loads(stripped)
                if isinstance(parsed, dict) and (
                    "inputs" in parsed or "input" in parsed
                ):
                    return run_stdin_stdout_tests(code, parsed, timeout)
                if isinstance(parsed, list) and parsed:
                    if isinstance(parsed[0], dict) and (
                        "input" in parsed[0] or "inputs" in parsed[0]
                    ):
                        return run_stdin_stdout_tests(code, parsed, timeout)
                    return run_assert_tests(code, parsed, timeout)
            except json.JSONDecodeError:
                pass
        return run_assert_tests(code, stripped, timeout)

    # List of strings -> assert mode; list of dicts -> stdin/stdout mode.
    if isinstance(test_cases, list) and test_cases:
        if isinstance(test_cases[0], dict):
            return run_stdin_stdout_tests(code, test_cases, timeout)
        return run_assert_tests(code, test_cases, timeout)

    if isinstance(test_cases, dict):
        return run_stdin_stdout_tests(code, test_cases, timeout)

    return {
        "passed": False, "total": 0, "passed_count": 0,
        "failed_count": 0, "error": "Could not determine test format",
        "details": [],
    }


def evaluate_code_response(
    response: str,
    test_cases: Any,
    timeout: float = 10.0,
    test_type: Optional[str] = None,
) -> Dict[str, Any]:
    """End-to-end evaluation: extract code -> check syntax -> run tests."""

    def _result(code, syntax_valid, syntax_error, test_result,
                format_reward, answer_reward, partial_reward):
        return {
            "code": code, "syntax_valid": syntax_valid,
            "syntax_error": syntax_error, "test_result": test_result,
            "format_reward": format_reward, "answer_reward": answer_reward,
            "partial_reward": partial_reward, "reward": answer_reward,
        }

    code = extract_code_from_response(response)
    if code is None:
        return _result(None, False, "No code found in response",
                       None, 0.0, 0.0, 0.0)

    syntax_ok, syntax_err = check_syntax(code)
    if not syntax_ok:
        return _result(code, False, syntax_err, None, 0.0, 0.0, 0.0)

    test_result = run_tests(code, test_cases, timeout=timeout, test_type=test_type)
    total = test_result.get("total", 0)
    passed_count = test_result.get("passed_count", 0)
    all_passed = test_result.get("passed", False)
    partial = (passed_count / total) if total > 0 else 0.0
    answer = 1.0 if all_passed else 0.0

    return _result(code, True, None, test_result, 1.0, answer, partial)
