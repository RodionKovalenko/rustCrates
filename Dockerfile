# Stage 1: Build the application
FROM rust:latest AS builder

WORKDIR /app

# Install system dependencies for building (openssl, zstd, xdo, etc.)
RUN apt-get update && apt-get install -y \
    libssl-dev \
    pkg-config \
    libxdo-dev \
    libfreetype6-dev \
    libfontconfig1-dev \
    libzstd-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy full source code
COPY . .

# Build in release mode
RUN cargo build --release

# Find the produced binary name
# We'll rename it to a known name (wav-transformer) in the next stage

# Stage 2: Runtime image with GLIBC >= 2.35 (to match Rust latest build)
FROM debian:bookworm-slim

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    git \
    libxdo3 \
    libfreetype6 \
    libfontconfig1 \
    libzstd1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ⚠️ Rename the built binary to a consistent name (wav-transformer)
# You MUST match this to your actual compiled binary name (from Cargo.toml [package].name)
# For example: npoa-wav-transformer => wav-transformer

COPY --from=builder /app/target/release/npoa-wav-transformer ./wav-transformer

# Copy additional resources
COPY --from=builder /app/templates ./templates
COPY --from=builder /app/STORAGE ./STORAGE
COPY --from=builder /app/datasets ./datasets
COPY --from=builder /app/src/neural_networks/tokenizers/gtp_neox_tokenizer.json ./src/neural_networks/tokenizers/gtp_neox_tokenizer.json

# Create a non-root user
RUN useradd -m appuser
RUN chown -R appuser:appuser /app

EXPOSE 7860

USER appuser

# Run the app
ENTRYPOINT ["./wav-transformer"]
CMD []
