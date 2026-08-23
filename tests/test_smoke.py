def test_package_root_imports() -> None:
    import amnesiac

    expected = ["AmnesiacError", "ConfigurationError", "Doc"]
    assert amnesiac.__all__ == expected
    assert all(getattr(amnesiac, name) is not None for name in amnesiac.__all__)
