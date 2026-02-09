import ast
import atexit
import contextlib
import io
import json
import multiprocessing as mp
import os
import re
import resource
import signal
import subprocess
import sys
import textwrap
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
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
# Sandbox helpers
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


@contextlib.contextmanager
def _sandbox_limits(cpu_timeout: int, max_memory: int):
    """Set CPU alarm and memory limits for pool workers; restore on exit."""
    def _alarm_handler(_signum, _frame):
        raise TimeoutError("CPU time limit exceeded")

    prev = signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(cpu_timeout)
    old_rlimit = resource.getrlimit(resource.RLIMIT_AS)
    try:
        resource.setrlimit(resource.RLIMIT_AS, (max_memory, max_memory))
    except (ValueError, OSError):
        pass
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, prev)
        try:
            resource.setrlimit(resource.RLIMIT_AS, old_rlimit)
        except (ValueError, OSError):
            pass


# ---------------------------------------------------------------------------
# Pool worker (handles both single and batch tests)
# ---------------------------------------------------------------------------

def _pool_worker_batch(task: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Execute candidate code once, then run each test in the same namespace."""
    warnings.filterwarnings("ignore", category=SyntaxWarning)
    code = task["code"]
    tests = task["tests"]
    cpu_timeout = task.get("cpu_timeout", 10)
    max_memory = task.get("max_memory", 512 * 1024 * 1024)

    _globals: dict = {"__builtins__": __builtins__}
    _old = sys.stdout

    with _sandbox_limits(cpu_timeout, max_memory):
        try:
            exec(compile(code, "<candidate>", "exec"), _globals)
        except TimeoutError as e:
            return [_fail(str(e), timeout=True) for _ in tests]
        except Exception as e:
            return [_fail(str(e)) for _ in tests]
        finally:
            sys.stdout = _old

        results: List[Dict[str, Any]] = []
        for test in tests:
            buf = io.StringIO()
            sys.stdout = buf
            try:
                exec(compile(test, "<test>", "exec"), _globals)
                sys.stdout = _old
                results.append({"passed": True, "error": None,
                                "stdout": buf.getvalue(), "timeout": False})
            except TimeoutError as e:
                sys.stdout = _old
                results.append(_fail(str(e), timeout=True))
                results.extend(
                    _fail("Skipped (timeout)", timeout=True)
                    for _ in range(len(tests) - len(results)))
                return results
            except Exception as e:
                sys.stdout = _old
                results.append({"passed": False, "error": str(e),
                                "stdout": buf.getvalue(), "timeout": False})

    return results


class SandboxPool:
    """Persistent process pool that eliminates per-task interpreter startup."""

    _instance: Optional["SandboxPool"] = None

    def __init__(self, max_workers: Optional[int] = None):
        self._max_workers = max_workers or min(os.cpu_count() or 4, 8)
        self._pool: Optional[mp.pool.Pool] = None

    def _ensure_pool(self):
        if self._pool is None:
            ctx = mp.get_context("forkserver")
            self._pool = ctx.Pool(
                processes=self._max_workers, maxtasksperchild=50)

    @classmethod
    def get_instance(cls, max_workers: Optional[int] = None) -> "SandboxPool":
        if cls._instance is None:
            cls._instance = cls(max_workers)
        return cls._instance

    def execute_batch_asserts(self, code: str, tests: List[str],
                              timeout: float = 10.0,
                              cpu_timeout: int = 10) -> List[Dict[str, Any]]:
        self._ensure_pool()
        task = {"code": code, "tests": tests, "cpu_timeout": cpu_timeout}
        ar = self._pool.apply_async(_pool_worker_batch, (task,))
        try:
            return ar.get(timeout=timeout)
        except mp.TimeoutError:
            return [_fail("Execution timed out", timeout=True) for _ in tests]
        except Exception as e:
            return [_fail(f"Pool error: {e}") for _ in tests]

    def execute_parallel(self, tasks: List[Tuple[str, str]],
                         timeout: float = 10.0,
                         cpu_timeout: int = 10) -> List[Dict[str, Any]]:
        self._ensure_pool()
        async_results = []
        for code, test_code in tasks:
            t = {"code": code, "tests": [test_code], "cpu_timeout": cpu_timeout}
            async_results.append(self._pool.apply_async(_pool_worker_batch, (t,)))
        results = []
        for ar in async_results:
            try:
                results.append(ar.get(timeout=timeout)[0])
            except mp.TimeoutError:
                results.append(_fail("Execution timed out", timeout=True))
            except Exception as e:
                results.append(_fail(f"Pool error: {e}"))
        return results

    def shutdown(self):
        if self._pool is not None:
            self._pool.terminate()
            self._pool.join()
            self._pool = None

    def __del__(self):
        self.shutdown()


def _cleanup_pool():
    global _stdio_pool
    if SandboxPool._instance is not None:
        SandboxPool._instance.shutdown()
    if _stdio_pool is not None:
        _stdio_pool.shutdown(wait=False)
        _stdio_pool = None

atexit.register(_cleanup_pool)


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
try:
    exec(compile(_code, "<candidate>", "exec"), _globals)
except Exception as _e:
    _err = {{"passed": False, "error": str(_e), "stdout": "", "timeout": False}}
    print(json.dumps([_err] * len(_tests)))
    sys.exit(0)

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
    """Fallback: run all asserts in a single subprocess (no pool)."""
    script = _BATCH_ASSERT_TEMPLATE.format(
        code_repr=repr(code),
        tests_repr=repr(tests),
        max_memory=max_memory,
        cpu_timeout=cpu_timeout,
    )
    try:
        result = subprocess.run(
            [sys.executable, "-S", "-W", "ignore::SyntaxWarning", "-c", script],
            capture_output=True, text=True, timeout=timeout,
        )
        return json.loads(result.stdout.strip())
    except subprocess.TimeoutExpired:
        return [{"passed": False, "error": "Execution timed out",
                 "stdout": "", "timeout": True} for _ in tests]
    except Exception as e:
        return [{"passed": False, "error": f"Execution error: {e}",
                 "stdout": "", "timeout": False} for _ in tests]


def run_assert_tests(code: str, test_list: Union[str, List[str]],
                     timeout: float = 10.0,
                     use_pool: bool = True) -> Dict[str, Any]:
    """Run assertion-based tests (MBPP / KodCode style)."""
    if isinstance(test_list, str):
        test_list = [test_list]
    tests = [t.strip() for t in test_list]

    if use_pool:
        try:
            pool = SandboxPool.get_instance()
            details = pool.execute_batch_asserts(code, tests, timeout=timeout)
        except Exception:
            details = _run_batch_asserts_subprocess(code, tests, timeout=timeout)
    else:
        details = _run_batch_asserts_subprocess(code, tests, timeout=timeout)

    return _make_summary(details, len(tests))


def run_stdin_stdout_tests(
    code: str,
    test_cases: Union[str, Dict, List[Dict]],
    timeout: float = 10.0,
    use_pool: bool = True,
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

    # Improvement D: parallel execution via a persistent ProcessPoolExecutor.
    # stdin/stdout tests need real subprocess (for stdin piping), so we
    # parallelise the subprocess calls rather than using the SandboxPool.
    if use_pool and len(norm_inputs) > 1:
        executor = _get_stdio_pool()
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


_stdio_pool: Optional[ProcessPoolExecutor] = None


def _get_stdio_pool() -> ProcessPoolExecutor:
    """Return a persistent ProcessPoolExecutor for stdin/stdout tests."""
    global _stdio_pool
    if _stdio_pool is None:
        _stdio_pool = ProcessPoolExecutor(
            max_workers=min(os.cpu_count() or 4, 8))
    return _stdio_pool


def _run_stdin_stdout_single(
    code: str, stdin_data: str, expected_stdout: str, timeout: float
) -> Dict[str, Any]:
    """Execute *code* with *stdin_data* and compare stdout to *expected_stdout*."""
    try:
        result = subprocess.run(
            [sys.executable, "-S", "-W", "ignore::SyntaxWarning", "-c", code],
            input=stdin_data,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        actual = result.stdout.strip()
        expected = expected_stdout.strip()
        passed = (actual == expected)

        return {
            "passed": passed,
            "actual_output": actual,
            "expected_output": expected,
            "error": result.stderr.strip() if result.returncode != 0 else None,
            "timeout": False,
        }
    except subprocess.TimeoutExpired:
        return {
            "passed": False,
            "actual_output": "",
            "expected_output": expected_stdout.strip(),
            "error": "Execution timed out",
            "timeout": True,
        }
    except Exception as e:
        return {
            "passed": False,
            "actual_output": "",
            "expected_output": expected_stdout.strip(),
            "error": str(e),
            "timeout": False,
        }


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

    # String → try JSON parse first.
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

    # List of strings → assert mode; list of dicts → stdin/stdout mode.
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
    """End-to-end evaluation: extract code → check syntax → run tests."""

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
