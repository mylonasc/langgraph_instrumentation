# Store Guide

## Scope

This directory contains synchronous persistence protocols and implementations.
Stores depend only on neutral models, identifiers, and store-level protocols.
They must not import adapters, recorders, processors, exporters, framework
packages, or optional observability SDKs.

## Contract

- Every mutating operation is atomic. Validation or persistence failure must
  leave the previously visible trace unchanged.
- Trace metadata and identifiers are schema-independent neutral records.
- A span start may reopen an existing finalized trace for a delayed child.
- Reads return immutable snapshots with deterministic span, metric, and summary
  ordering. Equal timestamps must have schema-independent canonical tie-breaks.
- Deletion, configured eviction, and retention remove whole traces; no span,
  event, metric, or parent reference may survive separately.
- Concurrent operations must preserve the same invariants as serialized calls.
- `close` is bounded and idempotent. A lock-acquisition timeout returns `False`
  without closing the store. Every data operation after close raises
  `RuntimeError` before validating other arguments.
- Buffered stores may implement `FlushableTraceStore`; synchronously committed
  stores should not add a meaningless persistence flush.

## Implementations

Every implementation, including future SQLite or PostgreSQL stores, must run
the reusable store contract tests. Durable stores must use transactions for
multi-record changes, document isolation and retry behavior, version schema
migrations, and avoid exposing database-specific objects through public APIs.
