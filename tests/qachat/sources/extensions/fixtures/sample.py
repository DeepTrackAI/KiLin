"""Synthetic Python source used only for static extraction tests."""

VALUE = 1


def identity(func):
    return func


@identity
def calculate(value):
    return value + VALUE


async def fetch():
    return VALUE


class Example:
    name = "example"

    @staticmethod
    def describe():
        return Example.name

    async def compute(self):
        return calculate(VALUE)


raise RuntimeError("Fixture source must not be executed or imported.")
