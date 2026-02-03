"""
Jinja2 rendering utils, used to generate new strategy and configurations.
"""


def render_template(templatefile: str, arguments: dict) -> str:
    from jinja2 import Environment, PackageLoader, select_autoescape

    env = Environment(
        loader=PackageLoader("freqtrade", "templates"),
        autoescape=select_autoescape(["html", "xml"]),
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
