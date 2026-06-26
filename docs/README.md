# Documentation

Guides for setting up, using, operating, and hardening the ODELIA Viewer Platform. New here?
Start with the [project README](../README.md) and its [quick start](../README.md#quick-start).

## Set up

- [setup/configuration.md](setup/configuration.md) — key config files and the Hugging Face `HF_TOKEN`
- [setup/setup_chat.md](setup/setup_chat.md) — set up the Chat AI backend (Ollama or llama.cpp), in order
- [setup/updating.md](setup/updating.md) — back up, pull, rebuild, and restart an existing deployment

## Use

- [usage/](usage/) — upload studies, run AI, chat, and manage users (the day-to-day workflow)
- [usage/adding_new_users.md](usage/adding_new_users.md) — create Keycloak users
- [usage/adding_custom_models.md](usage/adding_custom_models.md) — integrate your own AI model

## Understand

- [architecture.md](architecture.md) — services, repo layout, and how a study flows through the stack
- [models/](models/) — bundled-model cards (MST, MedGemma, chat) and their inputs, classes, and limits

## Secure

- [security/](security/) — production hardening and restricting published ports to localhost

## Reference

- [licensing.md](licensing.md) — per-component licenses and why the assembled stack is research-only
- [support.md](support.md) — collecting logs for a support request
