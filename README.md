# Fish Recognition CLI

A Go CLI tool that uses biometric analysis to identify individual fish. Same fish = same ID, different fish = different ID.

## Features

- Identifies individual fish using biometric analysis of scars, markings, and distinctive features
- Generates unique identifiers for specific fish individuals
- Compares two fish images with similarity confidence scoring
- Outputs JSON with detailed biometric data
- Supports common image formats (JPG, PNG, GIF, BMP)

## Prerequisites

- Go 1.23+ (for building from source)

## Installation

1. Clone the repository
2. Build the CLI:
   ```bash
   go build -o fish-recognition .
   ```

## Usage

The CLI has two main commands:

### Identify Individual Fish

Generate a unique identifier for a specific fish individual:

```bash
./fish-recognition identify path/to/fish-image.jpg
```

This analyzes the fish's unique characteristics (scars, markings, distinctive features) and generates a biometric ID. The same individual fish will always produce the same ID.

### Compare Two Fish

Compare two fish images and get a similarity confidence score:

```bash
./fish-recognition compare fish1.jpg fish2.jpg
```

This provides a confidence score (0-100%) indicating how likely the two images show the same individual fish.

## Example Workflows

### Track Individual Fish Over Time

```bash
# First catch of a bass
./fish-recognition identify bass-march-2024.jpg

# Same bass caught again in June
./fish-recognition identify bass-june-2024.jpg

# If it's the same fish, both commands will produce identical IDs
```

### Verify Tournament Catches

```bash
# Compare two anglers' catches to ensure they're different fish
./fish-recognition compare angler1-bass.jpg angler2-bass.jpg
```

## How It Works

The tool uses biometric analysis to:
1. Detect and analyze fish-specific features
2. Identify unique markings, scars, and distinctive characteristics
3. Generate a consistent biometric fingerprint for each individual fish
4. Compare fingerprints to determine if two images show the same fish

## Supported Image Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- GIF (.gif)
- BMP (.bmp)

## Command Help

Get help for any command:

```bash
./fish-recognition --help
./fish-recognition identify --help
./fish-recognition compare --help
```

# Vibe coded with Claude