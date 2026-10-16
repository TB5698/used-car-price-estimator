"""Check a car description before it reaches the model.

Python, not the LLM, decides whether a car can be priced. validate_car returns
a Validation whose status is one of:
    ok            everything needed is present and in range, result.car is ready
    needs_info    something is missing or not understood, ask the user
                  (question_for builds the question)
    out_of_scope  the car is outside what the model was trained on
"""
from dataclasses import dataclass, field
from datetime import date

LAST_DATA_YEAR = 2022  # the listings were collected in 2021

# column order the model Pipeline expects
MODEL_COLUMNS = [
    "year", "manufacturer", "model", "odometer", "condition", "fuel",
    "title_status", "transmission", "drive", "type",
]

# the app asks for these before it prices a car (blank values make estimates unstable)
REQUIRED = ["year", "manufacturer", "model", "odometer", "type", "drive", "fuel", "transmission"]

# used when the user does not say, and told to the user as an assumption
DEFAULTS = {"title_status": "clean"}

# values the model was trained on (read from the fitted model, "unknown" left out)
ALLOWED = {
    "manufacturer": [
        "acura", "alfa-romeo", "aston-martin", "audi", "bmw", "buick", "cadillac",
        "chevrolet", "chrysler", "dodge", "ferrari", "fiat", "ford", "gmc",
        "harley-davidson", "honda", "hyundai", "infiniti", "jaguar", "jeep", "kia",
        "land rover", "lexus", "lincoln", "mazda", "mercedes-benz", "mercury", "mini",
        "mitsubishi", "morgan", "nissan", "pontiac", "porsche", "ram", "rover",
        "saturn", "subaru", "tesla", "toyota", "volkswagen", "volvo",
    ],
    "type": ["sedan", "suv", "pickup", "coupe", "hatchback", "wagon", "van",
             "mini-van", "convertible", "bus", "offroad", "other"],
    "drive": ["fwd", "rwd", "4wd"],
    "fuel": ["gas", "diesel", "hybrid", "electric", "other"],
    "transmission": ["automatic", "manual", "other"],
    "title_status": ["clean", "rebuilt", "salvage", "lien"],
    "condition": ["new", "like new", "excellent", "good", "fair", "salvage"],
}

# common ways people say the same thing
ALIASES = {
    "manufacturer": {"chevy": "chevrolet", "vw": "volkswagen", "mercedes": "mercedes-benz",
                     "benz": "mercedes-benz", "landrover": "land rover"},
    "type": {"truck": "pickup", "minivan": "mini-van", "crossover": "suv", "station wagon": "wagon"},
    "drive": {"4x4": "4wd", "four wheel drive": "4wd", "front wheel drive": "fwd",
              "rear wheel drive": "rwd"},
    "fuel": {"gasoline": "gas", "petrol": "gas"},
    "transmission": {"auto": "automatic", "stick": "manual", "standard": "manual"},
    "title_status": {"rebuilt title": "rebuilt", "salvage title": "salvage"},
}

LABELS = {
    "year": "year", "manufacturer": "make", "model": "model", "odometer": "mileage",
    "type": "body type", "drive": "drive", "fuel": "fuel", "transmission": "transmission",
    "condition": "condition", "title_status": "title status",
}

BLANKS = {"", "unknown", "none", "n/a", "na", "null", "not sure", "unsure"}


@dataclass
class Validation:
    status: str                      # "ok", "needs_info" or "out_of_scope"
    car: dict = None                 # all model columns, set only when status is "ok"
    missing: list = field(default_factory=list)
    problems: list = field(default_factory=list)
    assumptions: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


def _blank_to_none(value):
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in BLANKS:
        return None
    return value


def _to_number(value):
    """Return a float, or None if the value is not a plain number."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).replace(",", "").strip())
    except ValueError:
        return None


def _check_choice(name, value, problems, warnings):
    text = str(value).strip().lower()
    text = ALIASES.get(name, {}).get(text, text)
    if text == "awd" and name == "drive":
        warnings.append("Treated AWD as 4wd, the closest drive type in the data.")
        text = "4wd"
    if text not in ALLOWED[name]:
        options = "" if name == "manufacturer" else f" Options: {', '.join(ALLOWED[name])}."
        problems.append(f"I do not recognize '{value}' as a {LABELS[name]}.{options}")
        return None
    return text


def validate_car(raw, cfg):
    """Check one car description. cfg is the data section of config.yaml."""
    car, missing, problems, assumptions, warnings = {}, [], [], [], []
    out_of_scope = False

    for name in MODEL_COLUMNS:
        value = _blank_to_none(raw.get(name))

        if value is None:
            car[name] = None
            if name in REQUIRED:
                missing.append(name)
            elif name in DEFAULTS:
                car[name] = DEFAULTS[name]
                assumptions.append(f"Assuming the {LABELS[name]} is {DEFAULTS[name]}, since you did not say.")
            continue

        if name == "model":
            car[name] = str(value).strip().lower()
        elif name == "year":
            number = _to_number(value)
            if number is None or number != int(number):
                problems.append(f"The year '{value}' is not a whole number.")
                car[name] = None
            elif number < cfg["MIN_YEAR"]:
                out_of_scope = True
                problems.append(f"This estimator covers {cfg['MIN_YEAR']} and newer cars, "
                                f"so a {int(number)} is outside what it can price.")
                car[name] = None
            elif number > date.today().year + 1:
                problems.append(f"The year {int(number)} looks wrong. Which model year is it?")
                car[name] = None
            else:
                car[name] = int(number)
                if number > LAST_DATA_YEAR:
                    warnings.append("The training listings are from 2021, so estimates for "
                                    f"{int(number)} cars are less reliable.")
        elif name == "odometer":
            number = _to_number(value)
            if number is None:
                problems.append(f"The mileage '{value}' is not a number.")
                car[name] = None
            elif number < 0:
                problems.append("The mileage cannot be negative.")
                car[name] = None
            elif number > cfg["MAX_ODOMETER"]:
                out_of_scope = True
                problems.append(f"This estimator covers cars up to {cfg['MAX_ODOMETER']:,} miles, "
                                f"so {number:,.0f} miles is outside what it can price.")
                car[name] = None
            else:
                car[name] = number
        else:
            car[name] = _check_choice(name, value, problems, warnings)

    if out_of_scope:
        status = "out_of_scope"
    elif missing or problems:
        status = "needs_info"
        # a field the user gave but we could not read still needs asking again
        for name in MODEL_COLUMNS:
            if name in REQUIRED and car.get(name) is None and name not in missing:
                missing.append(name)
    else:
        status = "ok"

    return Validation(
        status=status,
        car=car if status == "ok" else None,
        missing=missing, problems=problems,
        assumptions=assumptions, warnings=warnings,
    )


def _describe(name):
    label = LABELS[name]
    if name in ALLOWED and name != "manufacturer":
        return f"{label} ({', '.join(ALLOWED[name])})"
    return label


def question_for(result):
    """Plain-language message for the user, or None when the car is ready."""
    if result.status == "ok":
        return None
    parts = list(result.problems)
    if result.status == "needs_info" and result.missing:
        parts.append("To price this car I still need: "
                     + ", ".join(_describe(name) for name in result.missing) + ".")
    return " ".join(parts)
