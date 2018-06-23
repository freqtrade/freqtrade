import logging


logger = logging.getLogger(__name__)


try:
    from plotly import tools
    from plotly.offline import plot
    import plotly.graph_objs as go
except ImportError:
    logger.exception("Module plotly not found \n Please install using `pip install plotly`")
    exit()
