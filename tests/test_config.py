from pathlib import Path

from claude_code_search.config import load_config


def test_load_defaults_when_file_missing(tmp_path: Path) -> None:
    cfg = load_config(tmp_path / "nope.toml")
    assert cfg.embedding.model == "BAAI/bge-small-en-v1.5"
    assert cfg.embedding.device == "auto"
    assert cfg.chunking.max_chars == 1500
    assert cfg.summarization.model == "gemma2:9b"
    assert cfg.search.top_n == 10


def test_file_values_override_defaults(tmp_path: Path) -> None:
    p = tmp_path / "config.toml"
    p.write_text(
        "[embedding]\n"
        'model = "foo/bar"\n'
        "[search]\n"
        "top_n = 3\n"
    )
    cfg = load_config(p)
    assert cfg.embedding.model == "foo/bar"
    assert cfg.search.top_n == 3
    assert cfg.chunking.max_chars == 1500


def test_paths_root_is_expanded(tmp_path: Path) -> None:
    p = tmp_path / "c.toml"
    p.write_text('[paths]\nroot = "~/xyz"\n')
    cfg = load_config(p)
    assert cfg.paths.root == str(Path("~/xyz").expanduser())


def test_resolve_db_path_relative_to_project_root(tmp_path: Path) -> None:
    cfg = load_config(tmp_path / "none.toml", project_root=tmp_path)
    assert Path(cfg.paths.db).is_absolute()
    assert Path(cfg.paths.db).parent == tmp_path / "data"
