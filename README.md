# psoul

A CLI-first foundation for managed Python sessions.

> **v0.0.x — early development.** psoul is public from the first commit
> but the API, CLI surface, and event schema are unstable. Breaking changes
> may land in any release until v0.1.0.

## Install

```bash
uv add psoul
```

Or try it without installing:

```bash
uvx psoul --help
```

## Quick start

```bash
psoul version        # print installed version
psoul doctor         # check your Python environment
```

## Development

psoul uses [uv](https://docs.astral.sh/uv/) for package management.
[just](https://just.systems/) provides shorthand dev commands but is
optional — the underlying `uv run` commands work fine on their own.

```bash
just          # format, lint, type-check, and test
just format   # auto-fix formatting and lint issues
just lint     # lint, format check, and type-check
just test     # run tests only
just snap     # update inline snapshots
```

CI runs on Ubuntu, macOS, and Windows across Python 3.12, 3.13, and 3.14.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

[MIT](LICENSE)
