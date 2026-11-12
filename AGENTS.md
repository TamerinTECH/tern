# Agents

Read [CLAUDE.md](CLAUDE.md) before you change this repository: it has the build and check commands, the
properties that must stay true, the map of the source and the list of what is not done. Humans use
[README.md](README.md).

The short version:

- Gemma 4 only. One tern process per machine at a time.
- After a change: `make test` in the three builds, the same token ids as before (`--ids`), `--head-check`
  for the fast output layer, and the docs updated with the measurement.
- No tokens, model files, machine addresses or logins in git: `make secrets`. Commit and push only when asked.

Developed by TamerinTECH GmbH, MIT licence ([LICENSE](LICENSE)).
