from __future__ import annotations

import hashlib
import io
import tarfile
from pathlib import Path

import pytest

from test.paraver_fixture import FixtureRelease, prepare_fixture


def _archive(path: Path, entries: dict[str, bytes]) -> str:
    with tarfile.open(path, mode="w:gz") as archive:
        for name, contents in entries.items():
            info = tarfile.TarInfo(name)
            info.size = len(contents)
            archive.addfile(info, io.BytesIO(contents))
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.integration
def test_fixture_download_checks_digest_before_extracting(tmp_path: Path) -> None:
    archive = tmp_path / "fixture.tar.gz"
    _archive(archive, {"paraver/trace.prv": b"trace"})
    release = FixtureRelease(archive.as_uri(), "0" * 64, "test")
    cache = tmp_path / "cache" / "paraver"

    with pytest.raises(RuntimeError, match="checksum mismatch"):
        prepare_fixture(release, cache)

    assert not cache.exists()


@pytest.mark.integration
def test_fixture_download_rejects_path_traversal_without_partial_cache(tmp_path: Path) -> None:
    archive = tmp_path / "fixture.tar.gz"
    digest = _archive(archive, {"paraver/trace.prv": b"trace", "paraver/../../escaped": b"bad"})
    release = FixtureRelease(archive.as_uri(), digest, "test")
    cache = tmp_path / "cache" / "paraver"

    with pytest.raises(RuntimeError, match="Unsafe path"):
        prepare_fixture(release, cache)

    assert not cache.exists()
    assert not (tmp_path / "escaped").exists()


@pytest.mark.integration
def test_fixture_cache_hit_does_not_download_again(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    archive = tmp_path / "fixture.tar.gz"
    digest = _archive(archive, {"paraver/trace.prv": b"trace"})
    release = FixtureRelease(archive.as_uri(), digest, "test")
    cache = tmp_path / "cache" / "paraver"
    fixture_directory = prepare_fixture(release, cache)
    assert (fixture_directory / "trace.prv").read_bytes() == b"trace"

    def unexpected_download(*_args: object) -> str:
        raise AssertionError("warm fixture cache must not download")

    monkeypatch.setattr("test.paraver_fixture._download_archive", unexpected_download)
    cached_fixture = prepare_fixture(release, cache)
    assert cached_fixture == fixture_directory
    assert (cached_fixture / "trace.prv").read_bytes() == b"trace"
