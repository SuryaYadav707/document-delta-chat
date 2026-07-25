"""CLI entrypoint (Typer) — backs `make run` and `make markup`.

  compare --a <pid> --b <pid> [--no-llm]  -> build comparison, write report
  markup  --comparison <id>               -> annotated PDF (bonus)
"""
from __future__ import annotations

import typer

app = typer.Typer(add_completion=False)


@app.command()
def compare(a: str = typer.Option(..., "--a", help="base PID"),
            b: str = typer.Option(..., "--b", help="revised PID"),
            no_llm: bool = typer.Option(False, "--no-llm")):
    """Ingest two PIDs -> canonical -> delta -> report -> artifacts."""
    from src.services.comparison import ComparisonService

    cmp = ComparisonService().create(a, b, use_llm=not no_llm)
    s = cmp.summary
    typer.echo(f"comparison: {cmp.comparison_id}")
    typer.echo(f"changes: {s['total']} — " +
               ", ".join(f"{k}: {v}" for k, v in sorted(s["by_change_type"].items())))
    typer.echo("by kind: " + ", ".join(f"{k}={v}" for k, v in sorted(s["by_kind"].items())))
    typer.echo(f"report:  {cmp.artifacts_dir}/report.md (+ .json/.html)")


@app.command()
def markup(comparison: str = typer.Option(..., "--comparison", help="comparison id")):
    """Overlay delta bboxes onto the sheets -> annotated PDF (bonus)."""
    from src.markup.overlay import render_markup
    typer.echo(f"markup written: {render_markup(comparison)}")


if __name__ == "__main__":
    app()
