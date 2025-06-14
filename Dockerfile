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

# Stage 2: Runtime image with GLIBC >= 2.35 (to match Rust latest build)
FROM debian:bookworm-slim

# Install minimal runtime dependencies including wget and git
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

# Copy the compiled binary from the builder stage
COPY --from=builder /app/target/release/wav-transformer .

# Copy any needed templates (for HTML rendering)
COPY --from=builder /app/templates ./templates
COPY --from=builder /app/STORAGE ./STORAGE
COPY --from=builder /app/datasets ./datasets
COPY --from=builder /app/src/neural_networks/tokenizers/gtp_neox_tokenizer.json ./src/neural_networks/tokenizers/gtp_neox_tokenizer.json

# Create a non-root user first
RUN useradd -m appuser

# Change ownership of /app to appuser
RUN chown -R appuser:appuser /app

# Expose the port the web server uses
EXPOSE 7860

# Switch to non-root user
USER appuser

# Start the application
ENTRYPOINT ["./wav-transformer"]
CMD []