from pathlib import Path
from unittest.mock import patch

from claude_code_search import cli
from claude_code_search.config import Config
from claude_code_search.indexer import Indexer
from claude_code_search.store import connect, init_schema
from tests.test_indexer import FakeEmbedder, _fresh_root


def test_stats_command_reports_zero_on_empty_db(tmp_path: Path, capsys) -> None:
    cfg_path = tmp_path / "config.toml"
    (tmp_path / "data").mkdir()
    cfg_path.write_text(
        f'[paths]\nroot = "{tmp_path}"\ndb = "{tmp_path / "data" / "x.db"}"\n'
    )
    rc = cli.main(["--config", str(cfg_path), "stats"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "files indexed:   0" in out
    assert "sessions:        0" in out


def test_search_command_prints_expected_session(tmp_path: Path, capsys) -> None:
    root = _fresh_root(tmp_path)
    db_path = tmp_path / "data" / "x.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = connect(db_path)
    init_schema(conn)
    Indexer(conn, Config(), embedder=FakeEmbedder()).run(root)
    conn.close()

    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(
        f'[paths]\nroot = "{root}"\ndb = "{db_path}"\n'
    )

    from tests.test_searcher import QueryAwareFake

    with patch.object(cli, "_make_embedder", return_value=QueryAwareFake()):
        rc = cli.main(
            ["--config", str(cfg_path), "search", "DAG parse failures"]
        )
    out = capsys.readouterr().out
    assert rc == 0
    assert "bbb" in out


def test_search_shortcut_without_subcommand(tmp_path: Path, capsys) -> None:
    root = _fresh_root(tmp_path)
    db_path = tmp_path / "data" / "x.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = connect(db_path)
    init_schema(conn)
    Indexer(conn, Config(), embedder=FakeEmbedder()).run(root)
    conn.close()

    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(
        f'[paths]\nroot = "{root}"\ndb = "{db_path}"\n'
    )

    from tests.test_searcher import QueryAwareFake

    with patch.object(cli, "_make_embedder", return_value=QueryAwareFake()):
        rc = cli.main(["--config", str(cfg_path), "DAG parse failures"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "bbb" in out
