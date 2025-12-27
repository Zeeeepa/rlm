from rlm.environments.base_env import IsolatedEnv
from rlm.core.types import REPLResult

from typing import Optional, Tuple
import time
import json
import textwrap

import modal


# =============================================================================
# Default Modal Image
# =============================================================================


def get_default_image() -> modal.Image:
    """
    Build a default Modal image with common libraries for data science,
    math, and general Python work.
    """
    return (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install(
            # Build essentials
            "build-essential",
            "git",
            "curl",
            "wget",
            # For scientific computing
            "libopenblas-dev",
            "liblapack-dev",
        )
        .pip_install(
            # Data science essentials
            "numpy>=1.26.0",
            "pandas>=2.1.0",
            "scipy>=1.11.0",
            # Math & symbolic computation
            "sympy>=1.12",
            # HTTP & APIs
            "requests>=2.31.0",
            "httpx>=0.25.0",
            # Data formats
            "pyyaml>=6.0",
            "toml>=0.10.2",
            # Utilities
            "tqdm>=4.66.0",
            "python-dateutil>=2.8.2",
            "regex>=2023.0.0",
        )
    )


# =============================================================================
# REPL Script (runs inside the sandbox)
# =============================================================================

# Python script that runs in the sandbox as a persistent REPL
_REPL_SCRIPT = textwrap.dedent(
    '''
import sys
import io
import json
import base64
import traceback
import pickle

# Persistent namespace
_globals = {"__builtins__": __builtins__, "__name__": "__main__"}
_locals = {}

def serialize_locals(locals_dict):
    """Serialize locals to a safe representation."""
    result = {}
    for k, v in locals_dict.items():
        if k.startswith("_"):
            continue
        try:
            result[k] = repr(v)
        except:
            result[k] = f"<{type(v).__name__}>"
    return result

def execute_code(code):
    """Execute code and return stdout, stderr, and locals."""
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    old_stdout, old_stderr = sys.stdout, sys.stderr
    
    try:
        sys.stdout = stdout_buf
        sys.stderr = stderr_buf
        
        combined = {**_globals, **_locals}
        exec(code, combined, combined)
        
        # Update locals with new variables
        for key, value in combined.items():
            if key not in _globals and not key.startswith("_"):
                _locals[key] = value
        
    except Exception as e:
        traceback.print_exc(file=stderr_buf)
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
    
    return {
        "stdout": stdout_buf.getvalue(),
        "stderr": stderr_buf.getvalue(),
        "locals": serialize_locals(_locals),
    }

# Main REPL loop - read JSON commands from stdin
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    
    try:
        cmd = json.loads(line)
        
        if cmd.get("type") == "execute":
            code = cmd.get("code", "")
            result = execute_code(code)
            print(json.dumps({"success": True, "result": result}), flush=True)
        
        elif cmd.get("type") == "ping":
            print(json.dumps({"success": True, "pong": True}), flush=True)
        
        elif cmd.get("type") == "exit":
            print(json.dumps({"success": True, "exiting": True}), flush=True)
            break
        
        else:
            print(json.dumps({"success": False, "error": f"Unknown command: {cmd.get('type')}"}), flush=True)
    
    except Exception as e:
        print(json.dumps({"success": False, "error": str(e)}), flush=True)
'''
)


class ModalREPL(IsolatedEnv):
    """
    Modal REPL environment that runs Python code in a Modal Sandbox.

    This provides true isolation - code executes on Modal's infrastructure,
    completely separate from the local machine. State is maintained via a
    persistent Python subprocess running inside the sandbox.
    """

    def __init__(
        self,
        app_name: str = "rlm-sandbox",
        image: Optional[modal.Image] = None,
        timeout: int = 600,
        lm_handler_address: Optional[Tuple[str, int]] = None,
        context_payload: Optional[dict | list | str] = None,
        setup_code: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize the Modal REPL environment.

        Args:
            app_name: Name for the Modal app (will create if missing).
            image: Optional Modal Image with dependencies. Defaults to basic Python.
            timeout: Sandbox execution timeout in seconds.
            lm_handler_address: (host, port) for LM Handler communication.
            context_payload: Initial context to load into environment.
            setup_code: Code to run during setup.
        """
        super().__init__(**kwargs)

        self.app_name = app_name
        self.timeout = timeout
        self.lm_handler_address = lm_handler_address

        # Use provided image or build the default
        self.image = image or get_default_image()

        # Will be set during setup
        self.app = None
        self.sandbox = None
        self.repl_process = None

        # Setup the environment
        self.setup()

        # Load context if provided
        if context_payload is not None:
            self.load_context(context_payload)

        # Run setup code if provided
        if setup_code:
            self.execute_code(setup_code)

    def setup(self):
        """Create the Modal app, sandbox, and start the REPL process."""
        # Get or create the Modal app
        self.app = modal.App.lookup(self.app_name, create_if_missing=True)

        # Create a new sandbox
        self.sandbox = modal.Sandbox.create(
            app=self.app,
            image=self.image,
            timeout=self.timeout,
        )

        # Start the persistent REPL process
        self.repl_process = self.sandbox.exec(
            "python",
            "-u",
            "-c",
            _REPL_SCRIPT,
        )

        # Verify the REPL is running with a ping
        self._send_command({"type": "ping"})

    def _send_command(self, cmd: dict) -> dict:
        """Send a command to the REPL process and get the response."""
        if self.repl_process is None:
            raise RuntimeError("REPL process not running")

        # Write command to stdin
        cmd_json = json.dumps(cmd) + "\n"
        self.repl_process.stdin.write(cmd_json.encode())
        self.repl_process.stdin.drain()

        # Read response from stdout (one line)
        response_line = self.repl_process.stdout.readline()

        if not response_line:
            raise RuntimeError("REPL process closed unexpectedly")

        return json.loads(response_line)

    def load_context(self, context_payload: dict | list | str):
        """Load context into the sandbox environment."""
        if isinstance(context_payload, str):
            # For string context, escape it properly
            escaped = context_payload.replace("\\", "\\\\").replace('"""', '\\"\\"\\"')
            context_code = f'context = """{escaped}"""'
        else:
            # For JSON context, serialize it
            context_json = json.dumps(context_payload)
            escaped_json = context_json.replace("\\", "\\\\").replace("'", "\\'")
            context_code = f"import json; context = json.loads('{escaped_json}')"

        # Execute the context loading code
        self.execute_code(context_code)

    def execute_code(self, code: str) -> REPLResult:
        """Execute code in the Modal sandbox and return result."""
        start_time = time.perf_counter()

        # Send execute command
        response = self._send_command(
            {
                "type": "execute",
                "code": code,
            }
        )

        execution_time = time.perf_counter() - start_time

        if not response.get("success"):
            return REPLResult(
                stdout="",
                stderr=response.get("error", "Unknown error"),
                locals={},
                execution_time=execution_time,
                rlm_calls=[],
            )

        result = response.get("result", {})

        return REPLResult(
            stdout=result.get("stdout", ""),
            stderr=result.get("stderr", ""),
            locals=result.get("locals", {}),
            execution_time=execution_time,
            rlm_calls=[],  # TODO: Implement LLM call tracking via tunnels
        )

    def cleanup(self):
        """Terminate the REPL process and sandbox."""
        if self.repl_process is not None:
            try:
                self._send_command({"type": "exit"})
            except Exception:
                pass
            self.repl_process = None

        if self.sandbox is not None:
            try:
                self.sandbox.terminate()
            except Exception:
                pass
            self.sandbox = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        return False

    def __del__(self):
        self.cleanup()
