"""Tiny compatibility subset used when the optional pytest package is absent."""


class _Mark:
    @staticmethod
    def parametrize(_name, values):
        def decorate(function):
            def wrapped():
                for value in values:
                    function(value)
            return wrapped
        return decorate


class _Raises:
    def __init__(self, exception, match):
        self.exception = exception
        self.match = match

    def __enter__(self):
        return self

    def __exit__(self, kind, value, _traceback):
        if kind is None or not issubclass(kind, self.exception):
            return False
        assert self.match in str(value)
        return True


class _Pytest:
    mark = _Mark()
    raises = staticmethod(lambda exception, match: _Raises(exception, match))


mark = _Pytest.mark
raises = _Pytest.raises
