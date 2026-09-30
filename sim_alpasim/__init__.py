"""sim-alpasim: the platform's sim MCP server over AlpaSim's traffic service.

Implements the shared sim protocol's ``sim_rollout`` verb (1.1) by driving
AlpaSim's CAT-K traffic world model in-process: hand it a recorded scene and
a replacement trajectory for one agent, get every other agent re-simulated in
reaction. The upstream workspace (NVlabs/alpasim) supplies the model; this
package supplies the episode-schema adapter, the rollout driver and the MCP
surface.
"""

__all__ = ["__version__"]
__version__ = "0.1.0"
