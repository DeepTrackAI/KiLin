"""Small temperature utilities used by the QAChat sandbox."""


def celsius_to_kelvin(celsius: float) -> float:
    """Convert a Celsius temperature to Kelvin."""

    return celsius + 273.15


def summarize_temperatures(values: list[float]) -> dict[str, float]:
    """Return the minimum, mean, and maximum in Kelvin."""

    kelvin = [celsius_to_kelvin(value) for value in values]
    return {
        "minimum": min(kelvin),
        "mean": sum(kelvin) / len(kelvin),
        "maximum": max(kelvin),
    }
