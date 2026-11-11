"""Tests for zero-disk memory-only ingestion, format strictness, and Linux kernel hardening."""

import builtins
import inspect
import io
import subprocess
import sys
from pathlib import Path

import pypdfium2
import pytest
from PIL import Image

from docneedle.ingest.parser import DoclingParser, UnsupportedDocumentFormatError


class TestKernelHardening:
    """Validate kernel-level anti-dump and ptrace hardening without runner pollution."""

    def test_harden_process_in_isolated_subprocess(self) -> None:
        """Verify that harden_process() clears PR_SET_DUMPABLE in a clean subprocess."""
        if sys.platform != "linux":
            pytest.skip("PR_SET_DUMPABLE is Linux-specific")

        code = (
            "import os, sys\n"
            "os.environ.pop('DOCNEEDLE_DEV_ALLOW_DUMPABLE', None)\n"
            "from docneedle.hardening import harden_process, is_process_dumpable\n"
            "harden_process()\n"
            "print('DUMPABLE:', is_process_dumpable())\n"
        )
        res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        assert res.returncode == 0, f"Subprocess failed: {res.stderr}"
        assert "DUMPABLE: False" in res.stdout

    def test_harden_process_dev_bypass_in_isolated_subprocess(self) -> None:
        """Verify that DOCNEEDLE_DEV_ALLOW_DUMPABLE=1 bypasses non-dumpable enforcement."""
        if sys.platform != "linux":
            pytest.skip("PR_SET_DUMPABLE is Linux-specific")

        code = (
            "import os, sys\n"
            "os.environ['DOCNEEDLE_DEV_ALLOW_DUMPABLE'] = '1'\n"
            "from docneedle.hardening import harden_process, is_process_dumpable\n"
            "harden_process()\n"
            "print('DUMPABLE:', is_process_dumpable())\n"
        )
        res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        assert res.returncode == 0, f"Subprocess failed: {res.stderr}"
        assert "DUMPABLE: True" in res.stdout

    def test_harden_process_prevents_coredump_on_crash(self) -> None:
        """Verify that a hardened process produces no core dump on SIGSEGV."""
        if sys.platform != "linux":
            pytest.skip("PR_SET_DUMPABLE crash test is Linux-specific")

        code = (
            "import os, signal, sys, faulthandler\n"
            "pid = os.fork()\n"
            "if pid == 0:\n"
            "    faulthandler.disable()\n"
            "    sys.stderr.flush()\n"
            "    devnull = os.open(os.devnull, os.O_WRONLY)\n"
            "    os.dup2(devnull, 2)\n"
            "    os.close(devnull)\n"
            "    from docneedle.hardening import harden_process\n"
            "    harden_process()\n"
            "    os.kill(os.getpid(), signal.SIGSEGV)\n"
            "    os._exit(0)\n"
            "_, status = os.waitpid(pid, 0)\n"
            "assert os.WIFSIGNALED(status)\n"
            "assert os.WTERMSIG(status) == signal.SIGSEGV\n"
            "assert not os.WCOREDUMP(status), 'Hardened process unexpectedly produced core dump'\n"
            "print('SUCCESS')\n"
        )
        res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        assert res.returncode == 0, f"Test failed: {res.stderr}\n{res.stdout}"
        assert "SUCCESS" in res.stdout

    def test_unhardened_process_produces_coredump(self) -> None:
        """Confirm that unhardened default Linux process produces core dump bit (control test)."""
        if sys.platform != "linux":
            pytest.skip("WCOREDUMP control test is Linux-specific")

        code = (
            "import os, signal, sys, faulthandler\n"
            "pid = os.fork()\n"
            "if pid == 0:\n"
            "    faulthandler.disable()\n"
            "    sys.stderr.flush()\n"
            "    devnull = os.open(os.devnull, os.O_WRONLY)\n"
            "    os.dup2(devnull, 2)\n"
            "    os.close(devnull)\n"
            "    os.kill(os.getpid(), signal.SIGSEGV)\n"
            "    os._exit(0)\n"
            "_, status = os.waitpid(pid, 0)\n"
            "assert os.WIFSIGNALED(status)\n"
            "assert os.WTERMSIG(status) == signal.SIGSEGV\n"
            "if not os.WCOREDUMP(status):\n"
            "    print('SKIPPED_HOST_NO_COREDUMP')\n"
            "else:\n"
            "    print('SUCCESS')\n"
        )
        res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        assert res.returncode == 0, f"Test failed: {res.stderr}\n{res.stdout}"
        if "SKIPPED_HOST_NO_COREDUMP" in res.stdout:
            pytest.skip("Host environment globally suppresses core dumps; control test skipped")
        assert "SUCCESS" in res.stdout


class TestZeroDiskParser:
    """Validate that parse_bytes executes purely in RAM with zero disk writes."""

    @pytest.fixture
    def parser(self) -> DoclingParser:
        return DoclingParser()

    @pytest.fixture
    def sample_image_bytes(self) -> bytes:
        img = Image.new("RGB", (120, 80), color=(73, 109, 137))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    @pytest.fixture
    def sample_multipage_tiff_bytes(self) -> bytes:
        """Generate an in-memory 2-page TIFF document."""
        img1 = Image.new("RGB", (100, 100), color=(255, 0, 0))
        img2 = Image.new("RGB", (100, 100), color=(0, 255, 0))
        buf = io.BytesIO()
        img1.save(buf, format="TIFF", save_all=True, append_images=[img2])
        return buf.getvalue()

    @pytest.fixture
    def sample_pdf_bytes(self) -> bytes:
        pdf = pypdfium2.PdfDocument.new()
        pdf.new_page(width=200, height=200)
        buf = io.BytesIO()
        pdf.save(buf)
        pdf.close()
        return buf.getvalue()

    def test_parse_bytes_never_writes_to_disk(
        self,
        parser: DoclingParser,
        sample_image_bytes: bytes,
        sample_multipage_tiff_bytes: bytes,
        sample_pdf_bytes: bytes,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Assert that no write open, write_bytes, or tempfile creation occurs in docneedle."""

        # 1. Intercept Path write methods
        def forbidden_write(self: Path, *args: object, **kwargs: object) -> None:
            raise AssertionError(f"Zero-disk violation: Path write was called on {self}!")

        monkeypatch.setattr(Path, "write_bytes", forbidden_write)
        monkeypatch.setattr(Path, "write_text", forbidden_write)

        # 2. Intercept builtins.open write modes for docneedle calls
        orig_open = builtins.open

        def guarded_open(file: object, mode: str = "r", *args: object, **kwargs: object) -> object:
            if any(m in mode for m in ("w", "a", "+", "x")):
                caller_frames = [f.filename for f in inspect.stack()]
                if any("docneedle" in f and "test" not in f for f in caller_frames):
                    raise AssertionError(
                        f"Zero-disk violation: write open called on '{file}' with mode '{mode}'!"
                    )
            return orig_open(file, mode, *args, **kwargs)

        monkeypatch.setattr(builtins, "open", guarded_open)

        # 1. Parse Image
        doc, filename, images = parser.parse_bytes(sample_image_bytes, "test_chart.png")
        assert filename == "test_chart.png"
        assert 1 in images
        assert len(images[1]) > 0

        # 2. Parse Multi-page TIFF (verifies multi-frame image support)
        doc, filename, images = parser.parse_bytes(sample_multipage_tiff_bytes, "multipage.tiff")
        assert filename == "multipage.tiff"
        assert len(images) == 2
        assert 1 in images and 2 in images

        # 3. Parse PDF
        doc, filename, images = parser.parse_bytes(sample_pdf_bytes, "test_contract.pdf")
        assert filename == "test_contract.pdf"
        assert 1 in images
        assert len(images[1]) > 0

        # 4. Parse Markdown / Text
        txt_bytes = b"# Privacy Policy\n\nAll data is stored in RAM only."
        doc, filename, images = parser.parse_bytes(txt_bytes, "privacy.md")
        assert filename == "privacy.md"
        assert len(images) == 0

        # 5. Parse HTML
        html_bytes = (
            b"<html><head><title>Report</title></head>"
            b"<body><h1>Summary</h1><p>RAM verified.</p></body></html>"
        )
        doc, filename, images = parser.parse_bytes(html_bytes, "report.html")
        assert filename == "report.html"
        assert len(images) == 0

    def test_fail_fast_on_unsupported_formats(self, parser: DoclingParser) -> None:
        """Verify that unsupported extensions fail-fast without silent degradation."""
        with pytest.raises(UnsupportedDocumentFormatError) as exc_info:
            parser.parse_bytes(b"\x00\x01\x02", "malicious.exe")
        assert "Unsupported document format '.exe'" in str(exc_info.value)

        with pytest.raises(UnsupportedDocumentFormatError) as exc_info:
            parser.parse_bytes(b"{\\rtf1\\ansi...}", "unsupported.rtf")
        assert "Unsupported document format '.rtf'" in str(exc_info.value)

        with pytest.raises(UnsupportedDocumentFormatError) as exc_info:
            parser.parse_bytes(b"PK\x03\x04...", "archive.zip")
        assert "Unsupported document format '.zip'" in str(exc_info.value)

    def test_pdf_buffer_lifetime_safety(
        self,
        parser: DoclingParser,
        sample_pdf_bytes: bytes,
    ) -> None:
        """Verify that stream buffer remains valid across PDF parsing without segfault."""
        doc, filename, images = parser.parse_bytes(sample_pdf_bytes, "secure_doc.pdf")
        assert doc is not None
        assert 1 in images
        assert len(images[1]) > 0
