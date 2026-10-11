import importlib.metadata
import esis


def test_versions() -> None:
    result = esis._caching.versions("msfc-ccd", "not-an-installed-package")
    assert result == {
        "msfc-ccd": importlib.metadata.version("msfc-ccd"),
        "not-an-installed-package": None,
    }
