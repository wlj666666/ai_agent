"""Project-root-relative path resolution must not depend on the process cwd."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from drivetest_agent.ui import paths as paths_module
from drivetest_agent.ui.paths import compute_project_root


class TestComputeProjectRoot:
    def test_returns_three_parents_above_the_given_module_file(self, tmp_path: Path) -> None:
        module_file = tmp_path / "repo" / "src" / "drivetest_agent" / "ui" / "paths.py"
        module_file.parent.mkdir(parents=True)
        module_file.touch()

        result = compute_project_root(module_file)

        assert result == (tmp_path / "repo").resolve()

    def test_uses_absolute_resolution_even_for_relative_input(self, tmp_path: Path) -> None:
        module_file = tmp_path / "repo" / "src" / "drivetest_agent" / "ui" / "paths.py"
        module_file.parent.mkdir(parents=True)
        module_file.touch()

        result = compute_project_root(Path(module_file.as_posix()))

        assert result.is_absolute()
        assert result == (tmp_path / "repo").resolve()


class TestModuleLevelConstants:
    def test_project_root_points_at_the_real_repository_root(self) -> None:
        expected_root = Path(__file__).resolve().parents[2]
        assert paths_module.PROJECT_ROOT == expected_root

    def test_knowledge_dir_and_examples_path_are_under_project_root(self) -> None:
        assert paths_module.KNOWLEDGE_DIR == paths_module.PROJECT_ROOT / "knowledge"
        assert paths_module.EXAMPLES_PATH == paths_module.PROJECT_ROOT / "examples" / "cases.json"

    def test_knowledge_dir_exists_on_disk(self) -> None:
        assert paths_module.KNOWLEDGE_DIR.is_dir()

    def test_examples_path_exists_on_disk(self) -> None:
        assert paths_module.EXAMPLES_PATH.is_file()

    def test_paths_resolve_correctly_even_when_process_cwd_changes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        reloaded = importlib.reload(paths_module)
        try:
            assert reloaded.KNOWLEDGE_DIR.is_dir()
            assert reloaded.EXAMPLES_PATH.is_file()
            assert reloaded.PROJECT_ROOT == Path(__file__).resolve().parents[2]
        finally:
            importlib.reload(paths_module)
