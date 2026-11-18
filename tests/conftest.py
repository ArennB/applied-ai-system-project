"""Shared test setup."""

import os
import tempfile

# The app calls setup_logging() on import, and the first call fixes the log
# file for the whole process. Point it at a throwaway folder before any test
# runs, so test runs never write into the real logs/pawpal.log.
os.environ.setdefault("PAWPAL_LOG_DIR", tempfile.mkdtemp(prefix="pawpal-test-logs-"))

# The app loads GEMINI_API_KEY from .env on startup. Tests control the key
# themselves (and must never use a real one), so turn that off.
os.environ["PAWPAL_SKIP_DOTENV"] = "1"
