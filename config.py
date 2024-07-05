import contextlib


class Config:
    enable_backprop: bool = True
    train: bool = True


@contextlib.contextmanager
def using_config(name, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)


def no_grad():
    return using_config("enable_backprop", False)


def test_mode():
    return using_config('train', False)
