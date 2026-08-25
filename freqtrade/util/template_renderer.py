"""
Jinja2 rendering utils, used to generate new strategy and configurations.
"""


def render_template(templatefile: str, arguments: dict) -> str:
    from jinja2 import Environment, PackageLoader, select_autoescape

    env = Environment(
        loader=PackageLoader("freqtrade", "templates"),
        autoescape=select_autoescape(["html", "xml"]),
        # Every strategy_subtemplates/*.j2 fragment is rendered standalone here and then
        # concatenated into base_strategy.py.j2 -- a fragment's own leading/trailing
        # blank lines are the ONLY way to control spacing at those concatenation points
        # (base_strategy.py.j2's `{{- ... }}` tags deliberately trim their own
        # surrounding template-literal whitespace). Jinja2 strips a template's trailing
        # newline by default (keep_trailing_newline=False), which silently broke that --
        # keep it, so each fragment's rendered value matches its .j2 source exactly.
        keep_trailing_newline=True,
    )
    template = env.get_template(templatefile)
    return template.render(**arguments)


def render_template_with_fallback(
    templatefile: str, templatefallbackfile: str, arguments: dict | None = None
) -> str:
    """
    Use templatefile if possible, otherwise fall back to templatefallbackfile
    """
    from jinja2.exceptions import TemplateNotFound

    if arguments is None:
        arguments = {}
    try:
        return render_template(templatefile, arguments)
    except TemplateNotFound:
        return render_template(templatefallbackfile, arguments)
