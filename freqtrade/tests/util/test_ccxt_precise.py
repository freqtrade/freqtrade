from freqtrade.util import FtPrecise


ws = FtPrecise("-1.123e-6")
ws = FtPrecise("-1.123e-6")
xs = FtPrecise("0.00000002")
ys = FtPrecise("69696900000")
zs = FtPrecise("0")


def test_FtPrecise():
    assert ys * xs == "1393.938"
    assert xs * ys == "1393.938"

    assert ys + xs == "69696900000.00000002"
    assert xs + ys == "69696900000.00000002"
    assert xs - ys == "-69696899999.99999998"
    assert ys - xs == "69696899999.99999998"
    assert xs / ys == "0"
    assert ys / xs == "3484845000000000000"

    assert ws * xs == "-0.00000000000002246"
    assert xs * ws == "-0.00000000000002246"

    assert ws + xs == "-0.000001103"
    assert xs + ws == "-0.000001103"

    assert xs - ws == "0.000001143"
    assert ws - xs == "-0.000001143"

    assert xs / ws == "-0.017809439002671415"
    assert ws / xs == "-56.15"

    assert zs * ws == "0"
    assert zs * xs == "0"
    assert zs * ys == "0"
    assert ws * zs == "0"
    assert xs * zs == "0"
    assert ys * zs == "0"

    assert zs + ws == "-0.000001123"
    assert zs + xs == "0.00000002"
    assert zs + ys == "69696900000"
    assert ws + zs == "-0.000001123"
    assert xs + zs == "0.00000002"
    assert ys + zs == "69696900000"

    assert abs(FtPrecise("-500.1")) == "500.1"
    assert abs(FtPrecise("213")) == "213"

    assert abs(FtPrecise("-500.1")) == "500.1"
    assert -FtPrecise("213") == "-213"

    assert FtPrecise("10.1") % FtPrecise("0.5") == "0.1"
    assert FtPrecise("5550") % FtPrecise("120") == "30"

    assert FtPrecise("-0.0") == FtPrecise("0")
    assert FtPrecise("5.534000") == FtPrecise("5.5340")

    assert min(FtPrecise("-3.1415"), FtPrecise("-2")) == "-3.1415"

    assert max(FtPrecise("3.1415"), FtPrecise("-2")) == "3.1415"

    assert FtPrecise("2") > FtPrecise("1.2345")
    assert not FtPrecise("-3.1415") > FtPrecise("-2")
    assert not FtPrecise("3.1415") > FtPrecise("3.1415")
    assert FtPrecise.string_gt("3.14150000000000000000001", "3.1415")

    assert FtPrecise("3.1415") >= FtPrecise("3.1415")
    assert FtPrecise("3.14150000000000000000001") >= FtPrecise("3.1415")

    assert not FtPrecise("3.1415") < FtPrecise("3.1415")

    assert FtPrecise("3.1415") <= FtPrecise("3.1415")
    assert FtPrecise("3.1415") <= FtPrecise("3.14150000000000000000001")

    assert FtPrecise(213) == "213"
    assert FtPrecise(-213) == "-213"
    assert str(FtPrecise(-213)) == "-213"
    assert FtPrecise(213.2) == "213.2"
    assert float(FtPrecise(213.2)) == 213.2
    assert float(FtPrecise(-213.2)) == -213.2
