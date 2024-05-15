import pytest

from freqtrade.util import render_template, render_template_with_fallback


def test_render_template_fallback():
    from jinja2.exceptions import TemplateNotFound

    with pytest.raises(TemplateNotFound):
        val = render_template(
            templatefile="subtemplates/indicators_does-not-exist.j2",
            arguments={},
        )

    val = render_template_with_fallback(
        templatefile="strategy_subtemplates/indicators_does-not-exist.j2",
        templatefallbackfile="strategy_subtemplates/indicators_minimal.j2",
    )
    assert isinstance(val, str)
    assert "if self.dp" in val
