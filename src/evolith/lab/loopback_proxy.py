"""Bounded byte relay: host loopback to one isolated model container."""

from __future__ import annotations

import argparse
import ipaddress
import select
import socket
import socketserver
import threading


def private_target(value: str) -> str:
    address = ipaddress.ip_address(value)
    if (not isinstance(address, ipaddress.IPv4Address) or not address.is_private
            or address.is_loopback or address.is_link_local or address.is_unspecified):
        raise ValueError("The target must be a private container IPv4 address.")
    return str(address)


class RelayHandler(socketserver.BaseRequestHandler):
    def handle(self):
        try:
            with socket.create_connection(self.server.target, timeout=10) as upstream:
                self.request.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                upstream.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                self.request.settimeout(10)
                upstream.settimeout(10)
                peers = {self.request: upstream, upstream: self.request}
                readers = list(peers)
                while readers:
                    ready, _, _ = select.select(readers, [], [], 300)
                    if not ready:
                        return
                    for source in ready:
                        data = source.recv(65536)
                        if data:
                            peers[source].sendall(data)
                        else:
                            readers.remove(source)
                            peers[source].shutdown(socket.SHUT_WR)
        except OSError:
            # Transport disconnects must not log request contents.
            return


class LoopbackRelay(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True
    request_queue_size = 32

    def __init__(self, port: int, target: tuple[str, int], connections: int = 16):
        self.target = target
        self.slots = threading.BoundedSemaphore(connections)
        super().__init__(("127.0.0.1", port), RelayHandler)

    def process_request(self, request, client_address):
        if not self.slots.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except RuntimeError:
            self.slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.slots.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--target", type=private_target, required=True)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("The loopback port must be between 1024 and 65535.")
    with LoopbackRelay(args.port, (args.target, 8000)) as server:
        print(f"Listening on 127.0.0.1:{args.port}; fixed upstream {args.target}:8000", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
