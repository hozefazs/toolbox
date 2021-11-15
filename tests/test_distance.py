from toolbox.lib import haversine


def test_correct_distance():
    assert round(haversine(48.865070, 2.380009, 48.235070, 2.393409)) == 70


def test_correct_type():
    assert type(haversine(48.865070, 2.380009, 48.235070, 2.393409)) == float
