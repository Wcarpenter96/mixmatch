"""
CLI for MixMatch - extract BPM, sections, and keys from local audio files.
"""

import click
from mixmatch.audio_extractor import extract


@click.group()
def main():
    """MixMatch: Extract audio metadata from local files."""
    pass


@main.command()
@click.argument("audio_path")
@click.option(
    "--json",
    "output_json",
    is_flag=True,
    help="Output as JSON to stdout"
)
@click.option(
    "-o", "--output",
    "output_file",
    type=click.Path(),
    help="Write JSON output to file"
)
def extract_cmd(audio_path: str, output_json: bool, output_file: str):
    """Extract BPM, sections, and keys from a local audio file."""
    import json

    try:
        click.echo(f"🎵 Analyzing audio file: {audio_path}\n")
        result = extract(audio_path)

        # Write to file if output path specified
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)
            click.echo(f"✅ Output written to: {output_file}\n")

        if output_json:
            click.echo(json.dumps(result, indent=2))
        elif not output_file:
            # Only print human-readable output if not writing to file
            click.echo(f"✅ BPM: {result['bpm']}")
            click.echo(f"✅ Key: {result['key']}\n")
            click.echo(f"📍 Sections:\n")

            for i, section in enumerate(result["sections"], 1):
                click.echo(
                    f"{i}. {section['label'].capitalize()} @ {section['start']:.1f}s ({section['key']})"
                )
            click.echo()

    except FileNotFoundError:
        click.echo(f"❌ File not found: {audio_path}", err=True)
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)


if __name__ == "__main__":
    main()
