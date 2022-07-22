from ccxt import Precise


ws = Precise('-1.123e-6')
ws = Precise('-1.123e-6')
xs = Precise('0.00000002')
ys = Precise('69696900000')
zs = Precise('0')


def test_precise():
    assert ys * xs == '1393.938'
    assert xs * ys == '1393.938'

    assert ys + xs == '69696900000.00000002'
    assert xs + ys == '69696900000.00000002'
    assert xs - ys == '-69696899999.99999998'
    assert ys - xs == '69696899999.99999998'
    assert xs / ys == '0'
    assert ys / xs == '3484845000000000000'

    assert ws * xs == '-0.00000000000002246'
    assert xs * ws == '-0.00000000000002246'

    assert ws + xs == '-0.000001103'
    assert xs + ws == '-0.000001103'

    assert xs - ws == '0.000001143'
    assert ws - xs == '-0.000001143'

    assert xs / ws == '-0.017809439002671415'
    assert ws / xs == '-56.15'

    assert zs * ws == '0'
    assert zs * xs == '0'
    assert zs * ys == '0'
    assert ws * zs == '0'
    assert xs * zs == '0'
    assert ys * zs == '0'

    assert zs + ws == '-0.000001123'
    assert zs + xs == '0.00000002'
    assert zs + ys == '69696900000'
    assert ws + zs == '-0.000001123'
    assert xs + zs == '0.00000002'
    assert ys + zs == '69696900000'

    assert abs(Precise('-500.1')) == '500.1'
    assert abs(Precise('213')) == '213'

    assert abs(Precise('-500.1')) == '500.1'
    assert -Precise('213') == '-213'

    assert Precise('10.1') % Precise('0.5') == '0.1'
    assert Precise('5550') % Precise('120') == '30'

    assert Precise('-0.0') == Precise('0')
    assert Precise('5.534000') == Precise('5.5340')

    assert min(Precise('-3.1415'), Precise('-2')) == '-3.1415'

    assert max(Precise('3.1415'), Precise('-2')) == '3.1415'

    assert Precise('2') > Precise('1.2345')
    assert not Precise('-3.1415') > Precise('-2')
    assert not Precise('3.1415') > Precise('3.1415')
    assert Precise.string_gt('3.14150000000000000000001', '3.1415')

    assert Precise('3.1415') >= Precise('3.1415')
    assert Precise('3.14150000000000000000001') >= Precise('3.1415')

    assert not Precise('3.1415') < Precise('3.1415')

    assert Precise('3.1415') <= Precise('3.1415')
    assert Precise('3.1415') <= Precise('3.14150000000000000000001')
