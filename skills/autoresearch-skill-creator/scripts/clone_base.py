#!/usr/bin/env python3
"""Start a writable, full-history Autoresearch clone at a verified commit."""

import argparse
import os
from pathlib import Path
import re
import subprocess


UPSTREAM = "https://github.com/uditgoenka/autoresearch.git"
BASE = "7bddd056dfd0b0ded8360588d8101b1ecb5c0baa"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dest", type=Path, help="new directory for the clone")
    parser.add_argument("--revision", default=BASE, help="full upstream commit SHA")
    args = parser.parse_args()
    revision = args.revision.lower()
    if not re.fullmatch("[0-9a-f]{40}", revision):
        parser.error("--revision requires a full 40-digit hexadecimal commit SHA")

    dest = args.dest.expanduser().absolute()
    try:
        dest.mkdir(parents=True)  # Atomically reserve the name, even for concurrent runs.
    except FileExistsError:
        parser.exit(1, f"Destination already exists: {dest}\n")
    except OSError as error:
        parser.exit(1, f"Cannot create {dest}: {error}\n")

    try:
        os.chdir(dest)
        # A Git hook can export paths to its caller's worktree and index.
        local_vars = subprocess.check_output(["git", "rev-parse", "--local-env-vars"], text=True)
        env = os.environ.copy()
        for name in local_vars.splitlines():
            env.pop(name, None)

        def git(*args):
            return subprocess.check_output(["git", *args], env=env, text=True).strip()

        git("clone", "--origin", "upstream", "--no-checkout", "--reject-shallow", UPSTREAM, ".")
        if git("cat-file", "-t", revision) != "commit":
            raise ValueError(f"{revision} is not a commit")
        git("cat-file", "-e", f"{revision}:LICENSE")
        git("checkout", "--no-track", "-B", "skill-work", revision)
        git("update-ref", "refs/autoresearch/base", revision)
    except (OSError, ValueError, subprocess.CalledProcessError, KeyboardInterrupt) as error:
        parser.exit(1, f"Clone setup incomplete at {dest}: {error}\nInspect this directory before retrying.\n")
    print(f"Created {dest} on skill-work at {revision}")


if __name__ == "__main__":
    main()
