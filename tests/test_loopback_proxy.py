from __future__ import annotations

import socket
import socketserver
import threading

import pytest

from evolith.lab.loopback_proxy import LoopbackRelay, private_target


@pytest.mark.parametrize("address", ["127.0.0.1", "0.0.0.0", "169.254.1.1", "8.8.8.8", "::1"])
def test_relay_rejects_targets_outside_private_container_ipv4(address):
    with pytest.raises(ValueError):
        private_target(address)


def test_relay_preserves_bytes_and_half_closed_streams():
    class Echo(socketserver.BaseRequestHandler):
        def handle(self):
            while data := self.request.recv(65536):
                self.request.sendall(data)

    with socketserver.ThreadingTCPServer(("127.0.0.1", 0), Echo) as backend:
        thread = threading.Thread(target=backend.serve_forever, daemon=True)
        thread.start()
        with LoopbackRelay(0, backend.server_address) as relay:
            forward = threading.Thread(target=relay.serve_forever, daemon=True)
            forward.start()
            try:
                assert relay.server_address[0] == "127.0.0.1"
                with socket.create_connection(relay.server_address, timeout=3) as client:
                    payload = b"data: streaming-test\n\n" * 1024
                    client.sendall(payload)
                    client.shutdown(socket.SHUT_WR)
                    received = bytearray()
                    while data := client.recv(65536):
                        received.extend(data)
                    assert received == payload
            finally:
                relay.shutdown()
                forward.join(timeout=3)
        backend.shutdown()
        thread.join(timeout=3)
