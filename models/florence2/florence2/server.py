#!/usr/bin/env python3
import argparse
from nimbro_vision_server.server_base import ServerBase

def main():
    parser = argparse.ArgumentParser(
        description="Run a nimbro_vision_server instance for florence2"
    )
    parser.add_argument("--host", default="0.0.0.0",
                        help="Interface to bind to")
    parser.add_argument("--port", type=int, default=8000,
                        help="Port to listen on")
    parser.add_argument("--token", default=None,
                        help="Bearer token for auth (optional)")
    parser.add_argument("--preload_flavor", default=None,
                        help="Flavor to preload (optional)")
    args = parser.parse_args()

    # Initialize the server
    server = ServerBase(
        model_family="florence2",
        host=args.host,
        port=args.port,
        token=args.token,
        preload_flavor=args.preload_flavor
    )

    server.run()

if __name__ == "__main__":
    main()
