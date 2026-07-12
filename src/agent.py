"""LiveKit CLI entrypoint for the voice agent."""

from dotenv import load_dotenv
from livekit.agents import cli

load_dotenv(".env.local")

from agent.worker import (  # noqa: E402
    configure_livekit_log_level,
    server,
)

configure_livekit_log_level()

if __name__ == "__main__":
    cli.run_app(server)
