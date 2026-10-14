from __future__ import annotations

import hashlib
import os
import shutil
import sys
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from urllib.error import URLError
from urllib.request import urlopen

RELEASE_VERSION = "v1.0.0"
ARCHIVE_URL = (
    "https://github.com/Alexandre425/paraver-test-fixture/releases/download/"
    "v1.0.0/paraver-fixture-v1.0.0.tar.gz"
)
ARCHIVE_SHA256 = "8df1c0638ff9ca32f637251f410ca58c4288de1ec393811a4701ae096c5076f3"
CACHE_DIRECTORY = Path(__file__).parent / ".cache" / "paraver"
ARCHIVE_ROOT = "paraver"


@dataclass(frozen=True)
class FixtureRelease:
    url: str
    sha256: str
    version: str


RELEASE = FixtureRelease(ARCHIVE_URL, ARCHIVE_SHA256, RELEASE_VERSION)


def _download_archive(url: str, destination: Path) -> str:
    digest = hashlib.sha256()
    try:
        with urlopen(url, timeout=60) as response, destination.open("wb") as archive:
            while chunk := response.read(1024 * 1024):
                digest.update(chunk)
                archive.write(chunk)
    except (OSError, URLError) as error:
        raise RuntimeError(f"Unable to download the Paraver fixture from {url}: {error}") from error
    return digest.hexdigest()


def _extract_archive(archive_path: Path, destination: Path) -> None:
    seen: set[PurePosixPath] = set()
    try:
        with tarfile.open(archive_path, mode="r:gz") as archive:
            for member in archive:
                path = PurePosixPath(member.name)
                if path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] != ARCHIVE_ROOT:
                    raise RuntimeError(f"Unsafe path in Paraver fixture archive: {member.name!r}")
                if path in seen:
                    raise RuntimeError(f"Duplicate path in Paraver fixture archive: {member.name!r}")
                seen.add(path)
                target = destination.joinpath(*path.parts)
                if not target.resolve().is_relative_to(destination.resolve()):
                    raise RuntimeError(f"Unsafe path in Paraver fixture archive: {member.name!r}")
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                if not member.isfile():
                    raise RuntimeError(f"Unsupported entry in Paraver fixture archive: {member.name!r}")
                source = archive.extractfile(member)
                if source is None:
                    raise RuntimeError(f"Unable to read Paraver fixture archive entry: {member.name!r}")
                target.parent.mkdir(parents=True, exist_ok=True)
                with source, target.open("xb") as output:
                    shutil.copyfileobj(source, output)
    except (OSError, tarfile.TarError) as error:
        raise RuntimeError(f"Unable to extract the Paraver fixture archive: {error}") from error


def prepare_fixture(
    manifest: FixtureRelease = RELEASE,
    cache_directory: Path = CACHE_DIRECTORY,
) -> Path:
    cache_directory.parent.mkdir(parents=True, exist_ok=True)
    marker = cache_directory / ".complete"
    fixture_directory = cache_directory / ARCHIVE_ROOT
    expected_marker = f"{manifest.version} {manifest.sha256}\n"
    if marker.is_file() and marker.read_text(encoding="ascii") == expected_marker and fixture_directory.is_dir():
        return fixture_directory

    with tempfile.TemporaryDirectory(prefix="paraver-fixture-", dir=cache_directory.parent) as temporary_name:
        temporary = Path(temporary_name)
        archive_path = temporary / "fixture.tar.gz"
        actual_sha256 = _download_archive(manifest.url, archive_path)
        if actual_sha256 != manifest.sha256:
            raise RuntimeError(
                f"Paraver fixture checksum mismatch: expected {manifest.sha256}, received {actual_sha256}"
            )
        staging = temporary / "cache"
        staging.mkdir()
        _extract_archive(archive_path, staging)
        extracted_fixture = staging / ARCHIVE_ROOT
        if not extracted_fixture.is_dir():
            raise RuntimeError(f"Paraver fixture archive is missing its {ARCHIVE_ROOT}/ root directory")
        (staging / ".complete").write_text(expected_marker, encoding="ascii")
        if cache_directory.exists():
            shutil.rmtree(cache_directory)
        os.replace(staging, cache_directory)
    return fixture_directory


def main() -> int:
    try:
        fixture_directory = prepare_fixture()
    except RuntimeError as error:
        print(f"Paraver fixture preparation failed: {error}", file=sys.stderr)
        return 1
    print(f"Paraver fixture ready at {fixture_directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
