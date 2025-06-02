# Stage 1: Build the application
FROM rust:latest AS builder

WORKDIR /app

# Install system dependencies for building
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

# Stage 2: Runtime image with GLIBC >= 2.35
FROM debian:bookworm-slim

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y \
    libc6 \
    libssl3 \
    ca-certificates \
    git \
    libxdo3 \
    libfreetype6 \
    libfontconfig1 \
    libzstd1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /app/target/release/wav-transformer ./wav-transformer

# Copy additional resources
COPY --from=builder /app/templates ./templates
COPY --from=builder /app/STORAGE ./STORAGE
COPY --from=builder /app/datasets ./datasets
COPY --from=builder /app/src/neural_networks/tokenizers/gtp_neox_tokenizer.json ./src/neural_networks/tokenizers/gtp_neox_tokenizer.json

# Create a non-root user and give ownership
RUN useradd -m appuser \
    && chown -R appuser:appuser /app

# Expose the port expected by Hugging Face
EXPOSE 7860

# Enable full Rust backtraces for better debugging
ENV RUST_BACKTRACE=full

# Switch to non-root user
USER appuser

# Run the app with `server` argument
ENTRYPOINT ["./wav-transformer"]
CMD ["server"]
