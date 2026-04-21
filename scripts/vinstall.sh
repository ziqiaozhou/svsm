#!/bin/bash
# SPDX-License-Identifier: MIT OR Apache-2.0
#
# Copyright (c) Microsoft Corporation
#
# Author: Ziqiao Zhou <ziqiaozhou@microsoft.com>
# A script to install verus tools
set -e
trap 'echo "Error at line $LINENO: $BASH_COMMAND"' ERR

# Verus release version and commit hash
VERUS_REV=f1166c42c3decd42c1cca2916ef2880d27cfb7d9
VERUS_RUST_VERSION=1.94.0

# Verusfmt version and commit hash
VERUSFMT_REV=beff2fa686d856d5e60df368fd027d94ead11ac5 # v0.5.7

# Z3 version and commit hash
VERUS_Z3_REV=a7b564cafe3b96c8a868388bc4b96b319facea44

# Install x86_64-unknown-none target for verus-compatible Rust version
export RUSTUP_TOOLCHAIN=$VERUS_RUST_VERSION
rustup target add x86_64-unknown-none --toolchain $RUSTUP_TOOLCHAIN

fetch_code() {
    url=$1
    repo_name=$(basename "$url" .git)
    commit=$2
    git clone --no-checkout --depth 1 "$url" "$TMPDIR/$repo_name"
    cd "$TMPDIR/$repo_name"
    git fetch --depth 1 origin $commit
    git checkout FETCH_HEAD -b $commit
}

# Install verus toolchain into your ~/.cargo/bin
install_verus_assets() {
    VERUS_ASSETS=(
        "verus"
        "rust_verify"
        "z3"
        "cargo-verus"
        "verus-root"
    )
    local src_dir=$1
    for asset in "${VERUS_ASSETS[@]}"; do
        echo "Installing $asset to ~/.cargo/bin/"
        mv "$src_dir/$asset" ~/.cargo/bin/
    done
}

# Skip building Verus from source if the correct version is already installed
if (verus --version | grep -q "${VERUS_VERSION:0:7}") &> /dev/null && ! $FORCE_INSTALL; then
    echo "Verus version ${VERUS_VERSION:0:7} already installed, skipping build."
    exit 0
fi

# Install verusfmt
cargo install --git https://github.com/verus-lang/verusfmt  --rev $VERUSFMT_REV

# Create a temporary directory
TMPDIR=$(mktemp -d)
VERUS_DIR=$TMPDIR/verus

# Fetch Verus source code
fetch_code https://github.com/verus-lang/verus.git $VERUS_REV

# Build and install Z3 from source if not exists or version is wrong
if ! (z3 --version | grep -q "$VERUS_Z3_VERSION") &> /dev/null || $FORCE_INSTALL; then
    fetch_code https://github.com/Z3Prover/z3 $VERUS_Z3_REV
    python3 scripts/mk_make.py
    cd build && make -j$(nproc)
    cp $TMPDIR/z3/build/z3 $VERUS_DIR/source
else
    echo "Z3 found and version matches $VERUS_Z3_VERSION, skipping build."
    cp $(which z3) $VERUS_DIR/source
fi

# Build and install Verus from source
source $VERUS_DIR/tools/activate
cd $VERUS_DIR/source
rustup component add rust-src rustc-dev llvm-tools-preview --toolchain $RUSTUP_TOOLCHAIN
vargo build --release --vstd-no-verify
install_verus_assets "$VERUS_DIR/source/target-verus/release"
rm -rf "$TMPDIR"