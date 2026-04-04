import logging
import os
import subprocess
from app.tools.base_tool import Tool, ToolResult

logger = logging.getLogger(__name__)

class TestRunnerTool(Tool):
    """
    Phase 4 Autonomous Diagnostics Loops.
    Detects project limits cleanly spanning testing scripts executing validation safely preventing infinite hallucinated structures.
    """
    name = "TestRunnerTool"
    description = "Executes test runners mapped safely against structural directories verifying logic bounds natively."

    def __init__(self, repo_path: str):
        self.repo_path = repo_path

    def _detect_runner(self) -> str:
        # Layer 1 Detection natively preventing assumed limits!
        if os.path.exists(os.path.join(self.repo_path, "package.json")):
            return "npm test"
        
        if (os.path.exists(os.path.join(self.repo_path, "pytest.ini")) or
            os.path.exists(os.path.join(self.repo_path, "tests")) or
            os.path.exists(os.path.join(self.repo_path, ".pytest_cache"))):
            return "pytest"
            
        if os.path.exists(os.path.join(self.repo_path, "Makefile")):
            return "make test"
            
        # Layer 2 Fallback natively securely mapped.
        return "pytest"
        
    def summarize_test_failure(self, raw_output: str) -> str:
        """Phase 5.5: Heuristically condense noisy testing streams capturing explicitly failing assertions!"""
        import re
        markers = ["FAILURES", "ERRORS", "Traceback (most recent call last):", "FAIL ", "AssertionError"]
        summary = []
        capture = False
        
        lines = raw_output.splitlines()
        for idx, line in enumerate(lines):
            if any(m in line for m in markers):
                capture = True
            if capture:
                summary.append(line)
                
        if summary:
            # We enforce ~100 lines limit!
            s_out = "\\n".join(summary[-100:])
            return s_out[-1500:] if len(s_out) > 1500 else s_out
            
        # Fallback to the tail natively preventing blowing context bounds
        return raw_output[-1500:]

    def execute_tests(self) -> ToolResult:
        runner_cmd = self._detect_runner()
        logger.info(f"Phase 4 Native Validation limits matched via: {runner_cmd}")
        
        try:
            # We enforce a timeout bounding infinite loops intrinsically!
            res = subprocess.run(
                runner_cmd.split(), 
                cwd=self.repo_path, 
                capture_output=True, 
                text=True, 
                timeout=60
            )
            
            if res.returncode == 0:
                logger.info("Autonomous execution Test sequences PASSED correctly.")
                return ToolResult(
                    tool=self.name,
                    action="execute_tests",
                    success=True,
                    data={"message": f"Tests passed! ({runner_cmd})", "output": res.stdout[:500]}
                )
            else:
                logger.warning(f"Autonomous execution Tests FAILED natively: returncode {res.returncode}")
                # We extract stderr/stdout directly feeding back into Critic loops
                err_dump = (res.stderr or res.stdout or "No explicit error thrown!").strip()
                condensed = self.summarize_test_failure(err_dump)
                return ToolResult(
                    tool=self.name,
                    action="execute_tests",
                    success=False,
                    error=condensed # Phase 5.5: Explicit condensed boundaries
                )
        except subprocess.TimeoutExpired:
            return ToolResult(
                tool=self.name,
                action="execute_tests",
                success=False,
                error=f"Timeout executing runner: {runner_cmd}"
            )
        except Exception as e:
            return ToolResult(
                tool=self.name,
                action="execute_tests",
                success=False,
                error=f"Unrecognized runner crash explicitly bounded: {e}"
            )
