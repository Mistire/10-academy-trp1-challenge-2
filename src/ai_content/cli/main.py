"""
AI Content CLI.

Command-line interface for the AI content generation package.
"""

import asyncio
import logging
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.logging import RichHandler

# Import providers to register them
from ai_content import providers  # noqa: F401
from ai_content.core import ProviderRegistry, GenerationResult
from ai_content.config import configure, get_settings
from ai_content.presets import (
    get_music_preset,
    get_video_preset,
    list_music_presets,
    list_video_presets,
)

app = typer.Typer(
    name="ai-content",
    help="AI Content Generation CLI",
    no_args_is_help=True,
)
console = Console()


def setup_logging(verbose: bool = False):
    """Configure logging with rich output."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
    )


@app.callback()
def main(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging"),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Config file path"),
):
    """AI Content Generation CLI."""
    setup_logging(verbose)
    if config:
        configure(config_path=config)


# === Music Commands ===


@app.command()
def music(
    prompt: Optional[str] = typer.Option(None, "--prompt", "-p", help="Music style prompt"),
    provider: str = typer.Option("lyria", "--provider", help="Provider: lyria, minimax"),
    style: Optional[str] = typer.Option(None, "--style", "-s", help="Preset style name"),
    duration: int = typer.Option(30, "--duration", "-d", help="Duration in seconds"),
    bpm: int = typer.Option(120, "--bpm", help="Beats per minute"),
    lyrics: Optional[Path] = typer.Option(None, "--lyrics", "-l", help="Lyrics file"),
    reference_url: Optional[str] = typer.Option(
        None, "--reference-url", "-r", help="Reference audio URL for style transfer (MiniMax only)"
    ),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file"),
    force: bool = typer.Option(
        False, "--force", "-f", help="Force generation even if duplicate exists"
    ),
):
    """Generate music with AI."""
    if not prompt and not style:
        console.print("[red]Error: Either --prompt or --style must be provided[/red]")
        raise typer.Exit(1)
    asyncio.run(
        _generate_music(
            prompt=prompt,
            provider=provider,
            style=style,
            duration=duration,
            bpm=bpm,
            lyrics_file=lyrics,
            reference_url=reference_url,
            output=output,
            force=force,
        )
    )


async def _generate_music(
    prompt: Optional[str],
    provider: str,
    style: Optional[str],
    duration: int,
    bpm: int,
    lyrics_file: Optional[Path],
    reference_url: Optional[str],
    output: Optional[Path],
    force: bool = False,
):
    """Async music generation with job tracking."""
    from ai_content.core.job_tracker import get_tracker, JobStatus
    import shlex

    tracker = get_tracker()

    # Apply preset if specified
    if style:
        try:
            preset = get_music_preset(style)
            prompt = preset.prompt
            bpm = preset.bpm
            console.print(f"[green]Using preset: {style}[/green]")
        except KeyError as e:
            console.print(f"[red]{e}[/red]")
            raise typer.Exit(1)

    # Read lyrics if provided
    lyrics = None
    if lyrics_file:
        if not lyrics_file.exists():
            console.print(f"[red]Lyrics file not found: {lyrics_file}[/red]")
            raise typer.Exit(1)
        lyrics = lyrics_file.read_text()
        console.print(f"[green]Loaded lyrics: {len(lyrics)} characters[/green]")

    # Check for duplicates (unless --force)
    if not force:
        existing = tracker.find_duplicate(
            prompt=prompt,
            provider=provider,
            content_type="music",
            lyrics=lyrics,
            reference_url=reference_url,
        )
        if existing:
            if existing.status == JobStatus.COMPLETED or existing.status == JobStatus.DOWNLOADED:
                console.print(f"[yellow]⚠️ Duplicate found (already completed)[/yellow]")
                console.print(f"   Job ID: {existing.id}")
                if existing.output_path:
                    console.print(f"   Output: {existing.output_path}")
                console.print("[cyan]Use --force to generate anyway[/cyan]")
                raise typer.Exit(0)
            elif existing.status in (JobStatus.QUEUED, JobStatus.PROCESSING):
                console.print(f"[yellow]⚠️ Duplicate found (still processing)[/yellow]")
                console.print(f"   Job ID: {existing.id}")
                console.print(f"   Status: {existing.status.value}")
                console.print(
                    f"[cyan]Check status: uv run ai-content music-status {existing.id}[/cyan]"
                )
                raise typer.Exit(0)

    # Handle reference URL for style transfer
    if reference_url:
        console.print("[cyan]Using reference audio for style transfer[/cyan]")
        if provider != "minimax":
            console.print("[yellow]Note: Reference audio is only supported by MiniMax[/yellow]")

    # Get provider
    try:
        music_provider = ProviderRegistry.get_music(provider)
    except KeyError:
        available = ProviderRegistry.list_music_providers()
        console.print(f"[red]Unknown provider: {provider}. Available: {available}[/red]")
        raise typer.Exit(1)

    console.print(f"[blue]Generating with {provider}...[/blue]")

    # Build command string for tracking
    cmd_parts = ["ai-content", "music", "--prompt", shlex.quote(prompt), "--provider", provider]
    if lyrics_file:
        cmd_parts.extend(["--lyrics", str(lyrics_file)])
    if reference_url:
        cmd_parts.extend(["--reference-url", reference_url])
    if output:
        cmd_parts.extend(["--output", str(output)])
    command_str = " ".join(cmd_parts)

    # Generate
    result = await music_provider.generate(
        prompt=prompt,
        bpm=bpm,
        duration_seconds=duration,
        lyrics=lyrics,
        reference_audio_url=reference_url,
        output_path=str(output) if output else None,
    )

    # Track job if we got a generation ID
    if result.generation_id:
        tracker.create_job(
            generation_id=result.generation_id,
            provider=provider,
            content_type="music",
            prompt=prompt,
            command=command_str,
            lyrics=lyrics,
            reference_url=reference_url,
            metadata={"bpm": bpm, "duration": duration},
        )
        console.print(f"[dim]Job tracked: {result.generation_id}[/dim]")

        # Update status based on result
        if result.success:
            tracker.update_status(
                result.generation_id,
                JobStatus.DOWNLOADED if result.file_path else JobStatus.COMPLETED,
                output_path=str(result.file_path) if result.file_path else None,
            )
        elif result.error and "timeout" in result.error.lower():
            tracker.update_status(result.generation_id, JobStatus.PROCESSING)
        else:
            tracker.update_status(result.generation_id, JobStatus.FAILED)

    _print_result(result)


# === Video Commands ===


@app.command()
def video(
    prompt: Optional[str] = typer.Option(None, "--prompt", "-p", help="Scene description"),
    provider: str = typer.Option("veo", "--provider", help="Provider: veo, kling"),
    style: Optional[str] = typer.Option(None, "--style", "-s", help="Preset style name"),
    aspect: str = typer.Option("16:9", "--aspect", "-a", help="Aspect ratio"),
    duration: int = typer.Option(5, "--duration", "-d", help="Duration in seconds"),
    image: Optional[Path] = typer.Option(None, "--image", "-i", help="First frame image"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file"),
    fast: bool = typer.Option(False, "--fast", help="Use faster, lower-quality model"),
):
    """Generate video with AI."""
    if not prompt and not style:
        console.print("[red]Error: Either --prompt or --style must be provided[/red]")
        raise typer.Exit(1)
    asyncio.run(
        _generate_video(
            prompt=prompt,
            provider=provider,
            style=style,
            aspect=aspect,
            duration=duration,
            image=image,
            output=output,
            use_fast_model=fast,
        )
    )


async def _generate_video(
    prompt: Optional[str],
    provider: str,
    style: Optional[str],
    aspect: str,
    duration: int,
    image: Optional[Path],
    output: Optional[Path],
    use_fast_model: bool = False,
):
    """Async video generation."""
    # Apply preset if specified
    if style:
        try:
            preset = get_video_preset(style)
            prompt = preset.prompt
            aspect = preset.aspect_ratio
            console.print(f"[green]Using preset: {style}[/green]")
        except KeyError as e:
            console.print(f"[red]{e}[/red]")
            raise typer.Exit(1)

    # Get provider
    try:
        video_provider = ProviderRegistry.get_video(provider)
    except KeyError:
        available = ProviderRegistry.list_video_providers()
        console.print(f"[red]Unknown provider: {provider}. Available: {available}[/red]")
        raise typer.Exit(1)

    console.print(f"[blue]Generating with {provider}...[/blue]")

    # Generate
    first_frame = None
    if image and image.exists():
        # For now, we'd need to upload the image somewhere
        console.print("[yellow]Note: Image-to-video requires image URL[/yellow]")

    result = await video_provider.generate(
        prompt=prompt,
        aspect_ratio=aspect,
        duration_seconds=duration,
        first_frame_url=first_frame,
        output_path=str(output) if output else None,
        use_fast_model=use_fast_model,
    )

    _print_result(result)


# === Utility Commands ===


@app.command()
def list_providers():
    """List all available providers."""
    console.print("\n[bold]Music Providers:[/bold]")
    for name in ProviderRegistry.list_music_providers():
        console.print(f"  • {name}")

    console.print("\n[bold]Video Providers:[/bold]")
    for name in ProviderRegistry.list_video_providers():
        console.print(f"  • {name}")

    console.print("\n[bold]Image Providers:[/bold]")
    for name in ProviderRegistry.list_image_providers():
        console.print(f"  • {name}")


@app.command()
def list_presets():
    """List all available presets."""
    console.print("\n[bold]Music Presets:[/bold]")
    for name in list_music_presets():
        preset = get_music_preset(name)
        console.print(f"  • {name}: {preset.mood} ({preset.bpm} BPM)")

    console.print("\n[bold]Video Presets:[/bold]")
    for name in list_video_presets():
        preset = get_video_preset(name)
        console.print(f"  • {name}: {preset.aspect_ratio}")


def _print_result(result: GenerationResult):
    """Print generation result."""
    if result.success:
        console.print(f"\n[bold green]✅ Success![/bold green]")
        console.print(f"   Provider: {result.provider}")
        if result.file_path:
            console.print(f"   File: {result.file_path}")
        if result.file_size_mb:
            console.print(f"   Size: {result.file_size_mb:.2f} MB")
        if result.duration_seconds:
            console.print(f"   Duration: {result.duration_seconds}s")
    else:
        console.print("\n[bold red]❌ Failed![/bold red]")
        console.print(f"   Error: {result.error}")


@app.command("music-status")
def music_status(
    generation_id: str = typer.Argument(..., help="Generation ID from previous request"),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output file (downloads if complete)"
    ),
):
    """Check MiniMax music generation status and optionally download."""
    asyncio.run(_check_music_status(generation_id, output))


async def _check_music_status(generation_id: str, output: Optional[Path]):
    """Check generation status and download if ready."""
    from ai_content.providers.aimlapi.client import AIMLAPIClient
    from ai_content.core.job_tracker import get_tracker, JobStatus

    client = AIMLAPIClient()
    tracker = get_tracker()
    console.print(f"[cyan]Checking status for: {generation_id}[/cyan]")

    try:
        status = await client.poll_status("/v2/generate/audio", generation_id)
        state = status.get("status") or status.get("state", "unknown")
        console.print(f"[blue]Status: {state}[/blue]")

        if state.lower() in ("completed", "done", "success"):
            console.print("[green]✅ Generation complete![/green]")

            # Extract audio URL
            audio_url = None
            for key in ["audio_url", "url", "output"]:
                if key in status:
                    val = status[key]
                    if isinstance(val, str) and val.startswith("http"):
                        audio_url = val
                        break
                    elif isinstance(val, dict):
                        audio_url = val.get("audio_url") or val.get("url")
                        if audio_url:
                            break
                    elif isinstance(val, list) and val:
                        audio_url = val[0].get("audio_url") or val[0].get("url")
                        if audio_url:
                            break

            if audio_url and output:
                console.print(f"[blue]Downloading to {output}...[/blue]")
                audio_data = await client.download_file(audio_url)
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_bytes(audio_data)
                console.print(f"[green]✅ Saved to {output}[/green]")
                console.print(f"   Size: {len(audio_data) / (1024 * 1024):.2f} MB")
                # Update job tracker
                tracker.update_status(generation_id, JobStatus.DOWNLOADED, str(output))
            elif audio_url:
                console.print(f"[cyan]Audio URL: {audio_url}[/cyan]")
                console.print("[yellow]Use --output to download[/yellow]")
                tracker.update_status(generation_id, JobStatus.COMPLETED)
            else:
                console.print("[yellow]Audio URL not found in response[/yellow]")
                console.print(f"Response: {status}")
                tracker.update_status(generation_id, JobStatus.COMPLETED)

        elif state.lower() in ("queued", "pending", "processing"):
            console.print("[yellow]Still processing. Check again later.[/yellow]")
            console.print(f"Run: uv run ai-content music-status {generation_id}")
            if state.lower() == "processing":
                tracker.update_status(generation_id, JobStatus.PROCESSING)
        elif state.lower() in ("failed", "error"):
            error = status.get("error") or status.get("message") or "Unknown error"
            console.print(f"[red]❌ Generation failed: {error}[/red]")
            tracker.update_status(generation_id, JobStatus.FAILED)
        else:
            console.print("[yellow]Unknown status. Full response:[/yellow]")
            console.print(f"{status}")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
    finally:
        await client.close()


# === Job Management Commands ===


@app.command("jobs")
def jobs(
    status: Optional[str] = typer.Option(
        None,
        "--status",
        "-s",
        help="Filter by status: queued, processing, completed, downloaded, failed",
    ),
    provider: Optional[str] = typer.Option(None, "--provider", "-p", help="Filter by provider"),
    limit: int = typer.Option(20, "--limit", "-l", help="Max number of results"),
):
    """List tracked generation jobs."""
    from ai_content.core.job_tracker import get_tracker, JobStatus
    from rich.table import Table

    tracker = get_tracker()

    # Parse status filter
    status_filter = None
    if status:
        try:
            status_filter = JobStatus(status.lower())
        except ValueError:
            valid = ", ".join(s.value for s in JobStatus)
            console.print(f"[red]Invalid status '{status}'. Valid: {valid}[/red]")
            raise typer.Exit(1)

    job_list = tracker.list_jobs(status=status_filter, provider=provider, limit=limit)

    if not job_list:
        console.print("[yellow]No jobs found[/yellow]")
        return

    table = Table(title=f"Generation Jobs ({len(job_list)} shown)")
    table.add_column("ID", style="cyan", max_width=20)
    table.add_column("Provider", style="blue")
    table.add_column("Type", style="green")
    table.add_column("Status", style="magenta")
    table.add_column("Created", style="dim")
    table.add_column("Output", style="yellow", max_width=30)

    for job in job_list:
        # Color-code status
        status_str = job.status.value
        if job.status == JobStatus.COMPLETED or job.status == JobStatus.DOWNLOADED:
            status_str = f"[green]{status_str}[/green]"
        elif job.status == JobStatus.FAILED:
            status_str = f"[red]{status_str}[/red]"
        elif job.status in (JobStatus.QUEUED, JobStatus.PROCESSING):
            status_str = f"[yellow]{status_str}[/yellow]"

        table.add_row(
            job.id[:18] + "..." if len(job.id) > 20 else job.id,
            job.provider,
            job.content_type,
            status_str,
            job.created_at.strftime("%m-%d %H:%M"),
            (job.output_path[:28] + "...")
            if job.output_path and len(job.output_path) > 30
            else (job.output_path or "-"),
        )

    console.print(table)


@app.command("jobs-stats")
def jobs_stats():
    """Show job statistics summary."""
    from ai_content.core.job_tracker import get_tracker
    from rich.panel import Panel

    tracker = get_tracker()
    stats = tracker.get_stats()

    # Build stats display
    lines = [
        f"[bold]Total Jobs:[/bold] {stats['total']}",
        "",
        "[bold]By Status:[/bold]",
    ]

    for status, count in stats["by_status"].items():
        if count > 0:
            color = (
                "green"
                if status in ("completed", "downloaded")
                else "yellow"
                if status in ("queued", "processing")
                else "red"
            )
            lines.append(f"  [{color}]{status}:[/{color}] {count}")

    if stats["by_provider"]:
        lines.append("")
        lines.append("[bold]By Provider:[/bold]")
        for provider, count in stats["by_provider"].items():
            lines.append(f"  [blue]{provider}:[/blue] {count}")

    if stats["by_type"]:
        lines.append("")
        lines.append("[bold]By Type:[/bold]")
        for content_type, count in stats["by_type"].items():
            lines.append(f"  [cyan]{content_type}:[/cyan] {count}")

    lines.append("")
    lines.append(f"[dim]Recent (24h): {stats['recent_24h']}[/dim]")

    console.print(Panel("\n".join(lines), title="📊 Job Statistics", border_style="blue"))


@app.command("jobs-sync")
def jobs_sync(
    job_id: Optional[str] = typer.Option(None, "--id", help="Sync specific job ID"),
    download: bool = typer.Option(False, "--download", "-d", help="Download completed jobs"),
):
    """Sync status for pending jobs from the API."""
    asyncio.run(_sync_jobs(job_id, download))


async def _sync_jobs(job_id: str | None, download: bool):
    """Sync pending jobs with API status."""
    from ai_content.core.job_tracker import get_tracker, JobStatus
    from ai_content.providers.aimlapi.client import AIMLAPIClient

    tracker = get_tracker()
    client = AIMLAPIClient()

    if job_id:
        # Sync specific job
        job = tracker.get_job(job_id)
        if not job:
            console.print(f"[red]Job not found: {job_id}[/red]")
            return
        jobs_to_sync = [job]
    else:
        # Get all pending jobs
        jobs_to_sync = tracker.get_pending_jobs()

    if not jobs_to_sync:
        console.print("[green]No pending jobs to sync[/green]")
        return

    console.print(f"[blue]Syncing {len(jobs_to_sync)} job(s)...[/blue]")

    for job in jobs_to_sync:
        try:
            # Only sync MiniMax jobs (other providers may have different APIs)
            if job.provider != "minimax":
                console.print(f"[dim]Skipping {job.id} (provider: {job.provider})[/dim]")
                continue

            status = await client.poll_status("/v2/generate/audio", job.id)
            state = status.get("status") or status.get("state", "unknown")

            console.print(f"[cyan]{job.id[:15]}...:[/cyan] {state}")

            if state.lower() in ("completed", "done", "success"):
                # Extract audio URL if available
                audio_url = None
                for key in ["audio_url", "url", "output"]:
                    if key in status:
                        val = status[key]
                        if isinstance(val, str) and val.startswith("http"):
                            audio_url = val
                            break

                if audio_url and download:
                    # Download the file
                    output_path = Path(f"output/{job.content_type}/job_{job.id[:8]}.mp3")
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    audio_data = await client.download_file(audio_url)
                    output_path.write_bytes(audio_data)
                    tracker.update_status(job.id, JobStatus.DOWNLOADED, str(output_path))
                    console.print(f"   [green]Downloaded to {output_path}[/green]")
                else:
                    tracker.update_status(job.id, JobStatus.COMPLETED)

            elif state.lower() in ("failed", "error"):
                tracker.update_status(job.id, JobStatus.FAILED)
            elif state.lower() in ("processing",):
                tracker.update_status(job.id, JobStatus.PROCESSING)

        except Exception as e:
            console.print(f"   [red]Error syncing {job.id}: {e}[/red]")

    await client.close()
    console.print("[green]Sync complete[/green]")


def cli():
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    cli()
