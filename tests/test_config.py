from trustmargin_exp import config as config_module


def test_executable_path_preserves_virtualenv_python_symlink(monkeypatch, tmp_path):
    virtualenv = tmp_path / ".venv"
    virtualenv.mkdir()
    target = tmp_path / "base-python"
    target.touch()
    interpreter = virtualenv / "python"
    interpreter.symlink_to(target)

    monkeypatch.setenv("RAG_PYTHON", str(interpreter))

    assert config_module._executable_path("RAG_PYTHON", interpreter) == interpreter
