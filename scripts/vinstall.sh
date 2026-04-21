#!/bin/bash
# SPDX-License-Identifier: MIT OR Apache-2.0
#
# Copyright (c) Microsoft Corporation
#
# Author: Ziqiao Zhou <ziqiaozhou@microsoft.com>
# A script to install verus tools
set -e
trap 'echo "Error at line $LINENO: $BASH_COMMAND"' ERR
VERUS_RELEASE=0.2026.04.12.f1166c4 # May need to update if vstd or other verus library changes
VERUS_RUST_VERSION=1.94.0
VERUSFMT_REV=beff2fa686d856d5e60df368fd027d94ead11ac5 # v0.5.7

# Install x86_64-unknown-none target for verus-compatible Rust version
export RUSTUP_TOOLCHAIN=$VERUS_RUST_VERSION
rustup target add x86_64-unknown-none --toolchain $RUSTUP_TOOLCHAIN

# Install verusfmt
cargo install --git https://github.com/verus-lang/verusfmt  --rev $VERUSFMT_REV

# Install verus toolchain
# Verus cannot be installed via cargo and its build is slow, so we download the prebuilt binaries
VERUS_ASSETS=(
    "verus"
    "rust_verify"
    "z3"
    "cargo-verus"
    "verus-root"
)

# Detect platform
ARCH=$(uname -m)
OS=$(uname -s)
case "$ARCH" in
    x86_64)        PLATFORM_ARCH="x86" ;;
    aarch64|arm64) PLATFORM_ARCH="arm64" ;;
    *) echo "Error: unsupported architecture: $ARCH" >&2; exit 1 ;;
esac
case "$OS" in
    Linux)  PLATFORM_OS="linux" ;;
    Darwin) PLATFORM_OS="macos" ;;
    *) echo "Error: unsupported OS: $OS" >&2; exit 1 ;;
esac

PLATFORM="${PLATFORM_ARCH}-${PLATFORM_OS}"
ZIPFILE="verus-${VERUS_RELEASE}-${PLATFORM}.zip"
DOWNLOAD_URL="https://github.com/verus-lang/verus/releases/download/release/${VERUS_RELEASE}/${ZIPFILE}"
TMPDIR=$(mktemp -d)

# Download Verus prebuilt into a tmp folder
curl -sfL "$DOWNLOAD_URL" -o "$TMPDIR/$ZIPFILE"
unzip -q "$TMPDIR/$ZIPFILE" -d "$TMPDIR"

# Move the extracted Verus assets to the final directory
for asset in "${VERUS_ASSETS[@]}"; do
    echo "Installing $asset to ~/.cargo/bin/"
    mv "$TMPDIR/verus-$PLATFORM/$asset" ~/.cargo/bin/
done
rm -rf "$TMPDIR"