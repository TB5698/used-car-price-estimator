from src.validate import MODEL_COLUMNS, question_for, validate_car

CFG = {"MIN_YEAR": 1990, "MAX_ODOMETER": 300000}

FULL = dict(
    year=2016, manufacturer="toyota", model="camry", odometer=90000,
    type="sedan", drive="fwd", fuel="gas", transmission="automatic",
)


def car(**overrides):
    d = dict(FULL)
    d.update(overrides)
    return d


def test_complete_car_is_ok():
    result = validate_car(car(), CFG)
    assert result.status == "ok"
    assert list(result.car) == MODEL_COLUMNS
    assert result.car["year"] == 2016
    assert result.car["condition"] is None


def test_roadster_and_cabriolet_count_as_convertible():
    for word in ("Roadster", "cabriolet", "cabrio"):
        result = validate_car(car(type=word), CFG)
        assert result.status == "ok"
        assert result.car["type"] == "convertible"


def test_common_phrasings_map_to_allowed_values():
    cases = [
        ("type", "Pickup Truck", "pickup"), ("type", "sport utility", "suv"),
        ("type", "hatch", "hatchback"), ("type", "saloon", "sedan"),
        ("type", "mini van", "mini-van"),
        ("drive", "front-wheel drive", "fwd"), ("drive", "Rear Wheel Drive", "rwd"),
        ("drive", "four-wheel drive", "4wd"),
        ("fuel", "EV", "electric"),
        ("transmission", "stick shift", "manual"),
        ("transmission", "Automatic Transmission", "automatic"),
        ("manufacturer", "Alfa Romeo", "alfa-romeo"),
        ("manufacturer", "Mercedes Benz", "mercedes-benz"),
        ("manufacturer", "Harley Davidson", "harley-davidson"),
    ]
    for field_name, said, expected in cases:
        result = validate_car(car(**{field_name: said}), CFG)
        assert result.status == "ok", (field_name, said)
        assert result.car[field_name] == expected, (field_name, said)


def test_all_wheel_drive_in_words_is_4wd_with_warning():
    result = validate_car(car(drive="all-wheel drive"), CFG)
    assert result.status == "ok"
    assert result.car["drive"] == "4wd"
    assert result.warnings


def test_title_status_defaults_to_clean_and_says_so():
    result = validate_car(car(), CFG)
    assert result.car["title_status"] == "clean"
    assert any("clean" in a for a in result.assumptions)


def test_given_title_status_is_kept():
    result = validate_car(car(title_status="Rebuilt"), CFG)
    assert result.car["title_status"] == "rebuilt"
    assert result.assumptions == []


def test_missing_required_fields_need_info():
    raw = car()
    del raw["drive"]
    raw["type"] = None
    result = validate_car(raw, CFG)
    assert result.status == "needs_info"
    assert result.car is None
    assert set(result.missing) == {"drive", "type"}


def test_unknown_and_blank_values_count_as_missing():
    result = validate_car(car(fuel="unknown", transmission="", drive="N/A"), CFG)
    assert result.status == "needs_info"
    assert set(result.missing) == {"fuel", "transmission", "drive"}


def test_messy_text_is_normalized():
    result = validate_car(car(manufacturer=" Chevy ", type="Truck", transmission="AUTO",
                              fuel="Gasoline", drive="4x4", model="  Silverado 1500 "), CFG)
    assert result.status == "ok"
    assert result.car["manufacturer"] == "chevrolet"
    assert result.car["type"] == "pickup"
    assert result.car["transmission"] == "automatic"
    assert result.car["fuel"] == "gas"
    assert result.car["drive"] == "4wd"
    assert result.car["model"] == "silverado 1500"


def test_unrecognized_category_asks_again_with_options():
    result = validate_car(car(fuel="steam"), CFG)
    assert result.status == "needs_info"
    assert "fuel" in result.missing
    assert "steam" in question_for(result)
    assert "diesel" in question_for(result)


def test_invalid_condition_needs_info_even_though_optional():
    result = validate_car(car(condition="mint"), CFG)
    assert result.status == "needs_info"
    assert "mint" in question_for(result)


def test_too_old_is_out_of_scope():
    result = validate_car(car(year=1985), CFG)
    assert result.status == "out_of_scope"
    assert "1990" in question_for(result)


def test_too_many_miles_is_out_of_scope():
    result = validate_car(car(odometer=400000), CFG)
    assert result.status == "out_of_scope"
    assert "300,000" in question_for(result)


def test_out_of_scope_beats_missing_fields():
    raw = car(year=1980)
    del raw["drive"]
    result = validate_car(raw, CFG)
    assert result.status == "out_of_scope"
    assert "still need" not in question_for(result)


def test_non_numeric_mileage_needs_info():
    result = validate_car(car(odometer="lots"), CFG)
    assert result.status == "needs_info"
    assert "odometer" in result.missing


def test_numbers_as_text_are_accepted():
    result = validate_car(car(year="2018", odometer="90,000"), CFG)
    assert result.status == "ok"
    assert result.car["year"] == 2018
    assert result.car["odometer"] == 90000.0


def test_newer_than_training_data_is_ok_with_warning():
    result = validate_car(car(year=2024), CFG)
    assert result.status == "ok"
    assert result.warnings


def test_awd_is_treated_as_4wd_with_warning():
    result = validate_car(car(drive="AWD"), CFG)
    assert result.car["drive"] == "4wd"
    assert any("AWD" in w for w in result.warnings)


def test_question_lists_what_is_missing():
    raw = car()
    del raw["type"]
    text = question_for(validate_car(raw, CFG))
    assert "body type" in text and "sedan" in text


def test_ready_car_has_no_question():
    assert question_for(validate_car(car(), CFG)) is None
