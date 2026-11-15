from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


def helper(tmp_path, *, existing=False, port="18080", verification_fails=False):
    project = tmp_path / "project"
    scripts = project / "scripts"
    scripts.mkdir(parents=True)
    source = Path(__file__).resolve().parents[1] / "scripts" / "serve.sh"
    script = scripts / "serve.sh"
    script.write_text(source.read_text())
    profile = tmp_path / "engine.json"
    profile.write_text(json.dumps({"config": {"tensor_parallel": 2}}))
    binary = tmp_path / "bin"
    binary.mkdir()
    docker = binary / "docker"
    docker.write_text(
        "#!/usr/bin/env python3\n"
        "import json,os,sys\n"
        "with open(os.environ['COMMAND_LOG'],'a') as f:\n"
        "    f.write(json.dumps(sys.argv[1:])+'\\n')\n"
        "if sys.argv[1:3] == ['container','inspect']:\n"
        "    sys.exit(0 if os.environ['EXISTING']=='1' else 1)\n"
        "if sys.argv[1] == 'exec':\n"
        "    sys.exit(1 if os.environ['VERIFICATION_FAILS']=='1' else 0)\n"
        "if sys.argv[1] == 'inspect':\n"
        "    print('172.30.0.2'); sys.exit(0)\n"
        "print('test-id')\n"
    )
    docker.chmod(0o755)
    log = tmp_path / "commands.jsonl"
    result = subprocess.run(
        ["bash", str(script), str(profile)], check=False, capture_output=True, text=True,
        env=os.environ | {"PATH": f"{binary}:{os.environ['PATH']}",
                          "COMMAND_LOG": str(log), "EXISTING": str(int(existing)),
                          "VERIFICATION_FAILS": str(int(verification_fails)),
                          "EVOLITH_PORT": port, "EVOLITH_GPUS": "4,5,6,7"},
    )
    commands = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    return result, commands


def test_service_publishes_only_loopback_on_an_internal_network_and_required_gpus(tmp_path):
    result, commands = helper(tmp_path)
    assert result.returncode == 0, result.stderr
    network = next(c for c in commands if c[:2] == ["network", "create"])
    assert "--internal" in network
    runs = [c for c in commands if c[0] == "run"]
    assert len(runs) == 2
    command, relay = runs
    assert "-p" not in command
    assert command[command.index("--gpus") + 1] == '"device=4,5"'
    assert "--privileged" not in command and "--network=host" not in command
    assert "EVOLITH_LOOPBACK_PUBLISH=1" in command
    assert relay[relay.index("--network") + 1] == "host"
    assert relay[relay.index("--port") + 1] == "18080"
    assert relay[relay.index("--target") + 1] == "172.30.0.2"
    assert "--gpus" not in relay and "--read-only" in relay
    verification = next(c for c in commands if c[0] == "exec")
    assert "evolith.lab.prepublication" in verification
    assert commands.index(command) < commands.index(verification) < commands.index(relay)


def test_restart_output_mismatch_never_starts_the_relay_and_cleans_only_created_ids(tmp_path):
    result, commands = helper(tmp_path, verification_fails=True)
    assert result.returncode != 0
    assert len([c for c in commands if c[0] == "run"]) == 1
    assert ["rm", "-f", "test-id"] in commands
    assert ["network", "rm", "evolith-qwen-internal"] in commands


def test_service_never_replaces_an_existing_container(tmp_path):
    result, commands = helper(tmp_path, existing=True)
    assert result.returncode == 2
    assert len(commands) == 1
    assert commands[0][:2] == ["container", "inspect"]


def test_service_rejects_non_loopback_port_expression_before_docker(tmp_path):
    result, commands = helper(tmp_path, port="0.0.0.0:18080")
    assert result.returncode == 2
    assert not commands
