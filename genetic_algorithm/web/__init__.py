"""
GA Web Dashboard — Real-time monitoring and control for the Genetic Algorithm engine.

Provides a FastAPI backend with WebSocket streaming and a React frontend
for visualizing evolution progress, inspecting strategies, and controlling runs.

Usage:
    # Start dashboard only (no immediate evolution)
    python genetic_algorithm/run_ga.py --dashboard-only

    # Start evolution with dashboard
    python genetic_algorithm/run_ga.py --config config.yaml --dashboard

    # Programmatic usage
    from genetic_algorithm.web.server import create_app
    app = create_app()
"""

__version__ = "0.1.0"
