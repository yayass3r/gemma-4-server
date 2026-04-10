# Gemma 4 E2B Instruct - Model Server
# OpenAI-compatible API for Gemma 4 on Railway

## Overview
This server runs Google's Gemma 4 E2B Instruct model (Q4_K_M quantized) as an OpenAI-compatible API.

## Endpoints
| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Server info |
| GET | `/health` | Health check |
| POST | `/v1/chat/completions` | Chat completions |
| POST | `/v1/chat/completions/stream` | Streaming chat |

## Environment Variables
| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8000` | Server port |
| `CTX_SIZE` | `4096` | Context window size |
| `MAX_TOKENS` | `2048` | Max generation tokens |
| `MODEL_URL` | HuggingFace GGUF URL | Model download URL |

## Deployment
Deployed on Railway with Docker.
