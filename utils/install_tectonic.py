#!/usr/bin/env python3
"""Install Tectonic LaTeX engine to ~/.local/bin

This script automates the installation of Tectonic, a self-contained LaTeX engine.
Run this after installing Python dependencies.
"""

import os
import subprocess
import sys
import urllib.request
from pathlib import Path


def check_tectonic_installed():
    """Check if tectonic is already installed and in PATH or common locations."""
    # First check if it's in PATH
    try:
        result = subprocess.run(
            ["tectonic", "--version"],
            capture_output=True,
            text=True,
            check=True
        )
        version = result.stdout.strip()
        print(f"✓ Tectonic is already installed and in PATH: {version}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # Check common installation locations
    common_locations = [
        Path.home() / ".local" / "bin" / "tectonic",
        Path("/usr/local/bin/tectonic"),
        Path("/usr/bin/tectonic"),
    ]

    for location in common_locations:
        if location.exists():
            try:
                result = subprocess.run(
                    [str(location), "--version"],
                    capture_output=True,
                    text=True,
                    check=True
                )
                version = result.stdout.strip()
                print(f"✓ Tectonic is already installed at {location}: {version}")
                if location.parent not in os.environ.get("PATH", "").split(":"):
                    print(f"⚠  Note: {location.parent} is not in your PATH")
                    print(f"   Add this to your ~/.bashrc: export PATH=\"{location.parent}:$PATH\"")
                return True
            except (subprocess.CalledProcessError, FileNotFoundError):
                pass

    return False


def install_tectonic():
    """Download and install Tectonic binary."""
    print("Installing Tectonic LaTeX engine...")

    # Create ~/.local/bin if it doesn't exist
    local_bin = Path.home() / ".local" / "bin"
    local_bin.mkdir(parents=True, exist_ok=True)

    # Download and run the installer script
    installer_url = "https://drop-sh.fullyjustified.net"

    print(f"Downloading Tectonic from {installer_url}...")

    try:
        # Download installer script
        with urllib.request.urlopen(installer_url) as response:
            installer_script = response.read().decode('utf-8')

        # Run installer with --yes flag
        result = subprocess.run(
            ["sh", "-s", "--", "--yes"],
            input=installer_script,
            capture_output=True,
            text=True,
            check=True
        )

        print(result.stdout)

        # Move tectonic to ~/.local/bin
        tectonic_binary = Path("./tectonic")
        if tectonic_binary.exists():
            target = local_bin / "tectonic"
            tectonic_binary.rename(target)
            target.chmod(0o755)
            print(f"✓ Tectonic installed to {target}")
        else:
            print("✗ Tectonic binary not found after installation")
            return False

        # Verify installation
        if check_tectonic_installed():
            print("\n✓ Tectonic installation successful!")
            print(f"\nMake sure {local_bin} is in your PATH.")
            print("Add this to your ~/.bashrc or ~/.zshrc:")
            print(f'  export PATH="$HOME/.local/bin:$PATH"')
            return True
        else:
            print(f"\n⚠ Tectonic installed but not in PATH. Add {local_bin} to your PATH.")
            return True

    except Exception as e:
        print(f"✗ Installation failed: {e}")
        return False


def main():
    """Main entry point."""
    print("="*60)
    print("Tectonic LaTeX Engine Installer")
    print("="*60 + "\n")

    if check_tectonic_installed():
        print("\nNothing to do!")
        return 0

    print("Tectonic not found. Installing...\n")

    if install_tectonic():
        return 0
    else:
        print("\n✗ Installation failed. Please install manually:")
        print("  curl --proto '=https' --tlsv1.2 -fsSL https://drop-sh.fullyjustified.net | sh")
        return 1


if __name__ == "__main__":
    sys.exit(main())
