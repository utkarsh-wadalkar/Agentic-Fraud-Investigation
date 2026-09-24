"""Command-line interface for data preparation, investigation, and validation."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from fraud_agent.config import Settings
from fraud_agent.data import prepare_duckdb, profile_source_dir
from fraud_agent.graph import DuckDBGraphStore, HashEmbedding, write_graph_assets
from fraud_agent.graph_loading import GRAPH_LOAD_FILES, prepare_graph_files
from fraud_agent.knowledge import read_markdown_chunks, vector_records
from fraud_agent.mcp_runtime import open_mcp_graph_store
from fraud_agent.models import InvestigationAnswer
from fraud_agent.narrative import AnthropicNarrator
from fraud_agent.policy import validate_policy_consistency
from fraud_agent.training import TabularFraudModel, train_from_duckdb
from fraud_agent.workflow import CaseTrigger, InvestigationAgent, run_batch


@dataclass(frozen=True)
class ValidationReport:
    valid_files: int
    errors: list[str]


def resolve_backend(settings: Settings, requested: str) -> str:
    if requested not in {"auto", "local", "mcp"}:
        raise ValueError(f"unsupported backend {requested}")
    if requested == "auto":
        return "mcp" if settings.tigergraph_ready else "local"
    if requested == "mcp" and not settings.tigergraph_ready:
        raise ValueError("MCP backend requires TG_HOST, TG_GRAPHNAME, and a TigerGraph credential")
    return requested


def load_triggers(path: Path) -> list[CaseTrigger]:
    with path.open(newline="", encoding="utf-8") as handle:
        triggers = [CaseTrigger.from_row(row) for row in csv.DictReader(handle)]
    return sorted(triggers, key=lambda item: (item.opened_at, item.case_id))


def validate_answer_directory(cases_dir: Path, expected_count: int = 20) -> ValidationReport:
    errors: list[str] = []
    valid = 0
    files = sorted(cases_dir.glob("HHG-*.json"))
    if len(files) != expected_count:
        errors.append(f"expected {expected_count} answer files, found {len(files)}")
    for path in files:
        try:
            answer = InvestigationAnswer.model_validate(
                json.loads(path.read_text(encoding="utf-8"))
            )
            if answer.case_id != path.stem:
                errors.append(f"{path.name}: case_id does not match filename")
                continue
            policy_errors = validate_policy_consistency(answer)
            if policy_errors:
                errors.extend(f"{path.name}: {error}" for error in policy_errors)
                continue
            valid += 1
        except (json.JSONDecodeError, ValidationError) as error:
            errors.append(f"{path.name}: {error}")
    return ValidationReport(valid_files=valid, errors=errors)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fraud-agent")
    subparsers = parser.add_subparsers(dest="command", required=True)

    profile = subparsers.add_parser("profile", help="Validate and count source files")
    profile.add_argument("--source", type=Path, default=Path("Drive Files"))

    prepare = subparsers.add_parser("prepare", help="Build the local analytical store")
    prepare.add_argument("--source", type=Path, default=Path("Drive Files"))
    prepare.add_argument("--database", type=Path, default=Path(".artifacts/fraud.duckdb"))

    train = subparsers.add_parser("train", help="Train the historical fraud classifier")
    train.add_argument("--source", type=Path, default=Path("Drive Files"))
    train.add_argument("--database", type=Path, default=Path(".artifacts/fraud.duckdb"))
    train.add_argument("--model", type=Path, default=Path(".artifacts/fraud-model.joblib"))

    assets = subparsers.add_parser("graph-assets", help="Render TigerGraph GSQL assets")
    assets.add_argument("--out", type=Path, default=Path("graph"))
    assets.add_argument("--graph-name", default="FraudGraph")

    graph_prepare = subparsers.add_parser(
        "graph-prepare", help="Export normalized CSV files for TigerGraph loading"
    )
    graph_prepare.add_argument("--source", type=Path, default=Path("Drive Files"))
    graph_prepare.add_argument(
        "--database", type=Path, default=Path(".artifacts/fraud.duckdb")
    )
    graph_prepare.add_argument("--out", type=Path, default=Path(".artifacts/graph-load"))

    deploy = subparsers.add_parser("graph-deploy", help="Deploy schema and queries through MCP")
    deploy.add_argument("--assets", type=Path, default=Path("graph"))
    deploy.add_argument(
        "--replace",
        action="store_true",
        help="Drop and recreate the configured graph before deployment",
    )

    graph_load = subparsers.add_parser(
        "graph-load", help="Upload normalized CSV files through the MCP loading job"
    )
    graph_load.add_argument(
        "--data-dir", type=Path, default=Path(".artifacts/graph-load")
    )
    graph_load.add_argument("--job-name", default="load_fraud_graph")

    graph_knowledge = subparsers.add_parser(
        "graph-knowledge", help="Load benchmark documentation into TigerGraph vector search"
    )
    graph_knowledge.add_argument(
        "--source", type=Path, default=Path("Drive Files/README.md")
    )

    batch = subparsers.add_parser("batch", help="Investigate every benchmark case")
    batch.add_argument("--source", type=Path, default=Path("Drive Files"))
    batch.add_argument("--database", type=Path, default=Path(".artifacts/fraud.duckdb"))
    batch.add_argument("--cases-dir", type=Path, default=Path("cases"))
    batch.add_argument("--model", type=Path, default=Path(".artifacts/fraud-model.joblib"))
    batch.add_argument("--backend", choices=["auto", "local", "mcp"], default="auto")
    batch.add_argument("--no-llm", action="store_true")

    investigate = subparsers.add_parser("investigate", help="Investigate one case")
    investigate.add_argument("case_id")
    investigate.add_argument("--source", type=Path, default=Path("Drive Files"))
    investigate.add_argument("--database", type=Path, default=Path(".artifacts/fraud.duckdb"))
    investigate.add_argument("--cases-dir", type=Path, default=Path("cases"))
    investigate.add_argument("--model", type=Path, default=Path(".artifacts/fraud-model.joblib"))
    investigate.add_argument("--backend", choices=["auto", "local", "mcp"], default="auto")
    investigate.add_argument("--no-llm", action="store_true")

    validate = subparsers.add_parser("validate", help="Validate generated answers")
    validate.add_argument("--cases-dir", type=Path, default=Path("cases"))
    validate.add_argument("--expected-count", type=int, default=20)
    return parser


def _ensure_database(source: Path, database: Path) -> None:
    if not database.is_file():
        prepare_duckdb(source, database)


def load_or_train_model(database: Path, model_path: Path) -> tuple[TabularFraudModel, int | None]:
    if model_path.is_file():
        return TabularFraudModel.load(model_path), None
    model, sample_count = train_from_duckdb(database)
    model.save(model_path)
    return model, sample_count


async def _run_investigations(
    source: Path,
    database: Path,
    cases_dir: Path,
    model_path: Path,
    backend: str,
    use_llm: bool,
    case_id: str | None = None,
) -> list[InvestigationAnswer]:
    _ensure_database(source, database)
    triggers = load_triggers(source / "case_pack.csv")
    if case_id:
        triggers = [trigger for trigger in triggers if trigger.case_id == case_id]
        if not triggers:
            raise KeyError(f"unknown case {case_id}")
    settings = Settings(source_dir=source, cases_dir=cases_dir)
    selected_backend = resolve_backend(settings, backend)
    narrator = (
        AnthropicNarrator.from_settings(settings) if use_llm and settings.anthropic_ready else None
    )
    probability_model, _ = load_or_train_model(database, model_path)
    if selected_backend == "local":
        local_store = DuckDBGraphStore(database)
        agent = InvestigationAgent(local_store, narrator, probability_model)
        return await run_batch(triggers, agent, cases_dir)
    async with open_mcp_graph_store(settings) as mcp_store:
        agent = InvestigationAgent(mcp_store, narrator, probability_model)
        return await run_batch(triggers, agent, cases_dir)


async def _deploy_graph(assets: Path, replace: bool = False) -> None:
    settings = Settings()
    async with open_mcp_graph_store(settings) as store:
        if replace:
            await store.drop_graph()
        await store.execute_gsql((assets / "schema.gsql").read_text(encoding="utf-8"))
        await store.configure_knowledge_vectors()
        for filename in ("queries.gsql", "loading.gsql"):
            await store.execute_gsql((assets / filename).read_text(encoding="utf-8"))


async def _load_graph(data_dir: Path, job_name: str = "load_fraud_graph") -> list[str]:
    missing = [filename for filename in GRAPH_LOAD_FILES if not (data_dir / filename).is_file()]
    if missing:
        raise FileNotFoundError(f"missing graph load files: {', '.join(missing)}")
    settings = Settings()
    loaded: list[str] = []
    async with open_mcp_graph_store(settings) as store:
        for filename in GRAPH_LOAD_FILES:
            await store.load_file(
                data_dir / filename,
                f"{Path(filename).stem}_file",
                job_name=job_name,
            )
            loaded.append(filename)
    return loaded


async def _load_knowledge(source: Path) -> int:
    records = vector_records(read_markdown_chunks(source), HashEmbedding())
    settings = Settings()
    async with open_mcp_graph_store(settings) as store:
        for start in range(0, len(records), 50):
            await store.upsert_knowledge_vectors(records[start : start + 50])
    return len(records)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "profile":
        print(profile_source_dir(args.source))
        return 0
    if args.command == "prepare":
        print(prepare_duckdb(args.source, args.database))
        return 0
    if args.command == "train":
        _ensure_database(args.source, args.database)
        model, sample_count = train_from_duckdb(args.database)
        model.save(args.model)
        print(f"trained {args.model} from {sample_count} labeled transactions")
        return 0
    if args.command == "graph-assets":
        for path in write_graph_assets(args.out, args.graph_name):
            print(path)
        return 0
    if args.command == "graph-prepare":
        _ensure_database(args.source, args.database)
        manifest = prepare_graph_files(args.database, args.out)
        print(json.dumps(manifest.counts, indent=2, sort_keys=True))
        return 0
    if args.command == "graph-deploy":
        asyncio.run(_deploy_graph(args.assets, replace=args.replace))
        print(f"deployed graph assets from {args.assets}")
        return 0
    if args.command == "graph-load":
        loaded = asyncio.run(_load_graph(args.data_dir, args.job_name))
        print(f"loaded {len(loaded)} graph file(s) from {args.data_dir}")
        return 0
    if args.command == "graph-knowledge":
        count = asyncio.run(_load_knowledge(args.source))
        print(f"loaded {count} knowledge chunk(s) from {args.source}")
        return 0
    if args.command in {"batch", "investigate"}:
        case_id = args.case_id if args.command == "investigate" else None
        answers = asyncio.run(
            _run_investigations(
                args.source,
                args.database,
                args.cases_dir,
                args.model,
                backend=args.backend,
                use_llm=not args.no_llm,
                case_id=case_id,
            )
        )
        print(f"generated {len(answers)} answer file(s) in {args.cases_dir}")
        return 0
    if args.command == "validate":
        report = validate_answer_directory(args.cases_dir, args.expected_count)
        print(json.dumps({"valid_files": report.valid_files, "errors": report.errors}, indent=2))
        return 1 if report.errors else 0
    Settings()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
