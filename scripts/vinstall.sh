#!/bin/bash
# SPDX-License-Identifier: MIT OR Apache-2.0
#
# Copyright (c) Microsoft Corporation
#
# Author: Ziqiao Zhou <ziqiaozhou@microsoft.com>
# A script to install verus tools
set -e
trap 'echo "Error at line $LINENO: $BASH_COMMAND"' ERR
VERUS_RELEASE=f1166c42c3decd42c1cca2916ef2880d27cfb7d9 # May need to update if vstd or other verus library changes
VERUS_RUST_VERSION=1.94.0
VERUSFMT_REV=beff2fa686d856d5e60df368fd027d94ead11ac5 # v0.5.7
VERUS_Z3_REV=4.12.5

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

fetch_code() {
    url=$1
    repo_name=$(basename "$url" .git)
    commit=$2
    git clone --no-checkout --depth 1 "$url" "$TMPDIR/$repo_name"
    cd "$TMPDIR/$repo_name"
    git fetch --depth 1 origin $commit
    git checkout FETCH_HEAD -b $commit
}

# Skip building Verus if the correct version of Verus is already installed
if (verus --version | grep -q "${VERUS_RELEASE:0:7}") &> /dev/null; then
    echo "Verus version ${VERUS_RELEASE:0:7} already installed, skipping build."
    exit 0
fi

# Create a temporary directory
TMPDIR=$(mktemp -d)
VERUS_DIR=$TMPDIR/verus
# Build and install Verus from source
fetch_code https://github.com/verus-lang/verus.git $VERUS_RELEASE
git checkout $VERUS_RELEASE
# Build and install Z3 from source if not exists or version is wrong
if ! (z3 --version | grep -q "$VERUS_Z3_REV") &> /dev/null; then
    fetch_code https://github.com/Z3Prover/z3 z3-$VERUS_Z3_REV
    python3 scripts/mk_make.py
    cd build && make -j$(nproc)
    cp $TMPDIR/z3/build/z3 $VERUS_DIR/source
else
    echo "Z3 found and version matches $VERUS_Z3_REV, skipping build."
    cp $(which z3) $VERUS_DIR/source
fi
source $VERUS_DIR/tools/activate
cd $VERUS_DIR/source
rustup component add rust-src rustc-dev llvm-tools-preview --toolchain $RUSTUP_TOOLCHAIN
vargo build --release

# Move the Verus and z3 binaries to the final installation directory
for asset in "${VERUS_ASSETS[@]}"; do
    echo "Installing $asset to ~/.cargo/bin/"
    mv "$VERUS_DIR/source/target-verus/release/$asset" ~/.cargo/bin/
done
rm -rf "$TMPDIR"
fi