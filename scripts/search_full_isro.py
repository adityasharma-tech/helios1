import typer
from rich.console import Console
from rich.table import Table
import numpy as np
import cv2
import torch
import os
from tqdm import tqdm

from src.data import ISROReader, normalize_16bit_to_8bit
from src.registration import FeatureMatcher, geometric_verification

console = Console()
app = typer.Typer()

@app.command()
def search(
    isro_img: str = typer.Option(..., help="Path to ISRO PDS4 .img file"),
    isro_xml: str = typer.Option(..., help="Path to ISRO PDS4 .xml file"),
    query_img: str = typer.Option(..., help="Path to query square PNG image"),
    window_size: int = typer.Option(5000, help="Size of the sliding window"),
    stride: int = typer.Option(4000, help="Stride of the sliding window (overlap)"),
    max_dim: int = typer.Option(1024, help="Max dimension to scale windows down for GPU"),
    method: str = typer.Option("disk", help="Feature extraction method (disk or sift)")
):
    """
    Search for a query image across the ENTIRE ISRO strip using a sliding window.
    Evaluates every region and sorts the best matches to the top.
    """
    if not os.path.exists(isro_img) or not os.path.exists(query_img):
        console.print("[red]Input files not found. Please check paths.[/red]")
        return
        
    console.print(f"[bold cyan]1. Reading Query Image[/bold cyan]")
    source_raw = cv2.imread(query_img, cv2.IMREAD_GRAYSCALE)
    if source_raw is None:
        console.print("[red]Could not read query image.[/red]")
        return
        
    def resize_for_gpu(img, max_dim):
        h, w = img.shape
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return img
        
    source_img = resize_for_gpu(source_raw, max_dim)
    console.print(f"Query Image loaded and resized to {source_img.shape} for GPU")
    
    # Load Neural Network Matcher once
    matcher = FeatureMatcher(method=method)
    
    results = []
    
    console.print(f"\n[bold cyan]2. Scanning Entire ISRO Image (Sliding Window)[/bold cyan]")
    with ISROReader(isro_img, isro_xml) as reader:
        h, w = reader.height, reader.width
        console.print(f"ISRO Total Shape: {h}x{w}")
        
        # Pre-calculate sliding window chunks
        chunks = []
        for y in range(0, h - window_size // 4, stride):
            for x in range(0, w - window_size // 4, stride):
                chunks.append((x, y))
                
        console.print(f"Total regions to search: {len(chunks)} windows")
        
        for x, y in tqdm(chunks, desc="Searching via Neural Matching"):
            # Ensure we don't go out of bounds
            y2 = min(y + window_size, h)
            x2 = min(x + window_size, w)
            
            # Quick slice via memory map
            patch_raw = np.array(reader.mm[y:y2, x:x2])
            
            # If patch is mostly black/empty (e.g. edge of strip), skip to save time
            if np.mean(patch_raw) < 10:
                continue
                
            patch_8bit = normalize_16bit_to_8bit(patch_raw)
            target_img = resize_for_gpu(patch_8bit, max_dim)
            
            try:
                # Extract features and match
                pts1, pts2, kp1, kp2, matches = matcher.extract_and_match(source_img, target_img)
                inlier_count = 0
                
                # Verify geometric consistency
                if len(matches) >= 4:
                    M, inliers = geometric_verification(pts1, pts2)
                    inlier_count = inliers.sum() if len(inliers) > 0 else 0
                
                # Only record regions that actually have verifiable matches
                if inlier_count > 0:
                    results.append({
                        "x": x,
                        "y": y,
                        "cx": x + (x2 - x) // 2,
                        "cy": y + (y2 - y) // 2,
                        "inliers": inlier_count,
                        "matches": len(matches)
                    })
            except Exception as e:
                # Ignore random OpenCV/Cuda errors on edge cases
                pass
                
    if not results:
        console.print("[red]No matches found anywhere in the entire ISRO image.[/red]")
        return
        
    # Sort results by highest inlier count (Best structural match)
    results.sort(key=lambda r: r['inliers'], reverse=True)
    
    console.print(f"\n[bold cyan]3. Top Matches Found[/bold cyan]")
    table = Table(box=None)
    table.add_column("Rank", style="bold")
    table.add_column("Center Point (CX, CY)", style="cyan")
    table.add_column("Bounding Box (X, Y)")
    table.add_column("Inliers", justify="right")
    table.add_column("Matches", justify="right")
    
    for i, res in enumerate(results[:15]): # Show top 15
        table.add_row(
            f"#{i+1}",
            f"{res['cx']}, {res['cy']}",
            f"{res['x']}, {res['y']}",
            f"[green]{res['inliers']}[/green]",
            f"{res['matches']}"
        )
        
    console.print(table)
    
    best = results[0]
    console.print(f"\n[bold green]To visualize the #1 BEST match, copy-paste this command:[/bold green]")
    console.print(f"python main.py register \\")
    console.print(f"  --isro-img \"{isro_img}\" \\")
    console.print(f"  --isro-xml \"{isro_xml}\" \\")
    console.print(f"  --query-img \"{query_img}\" \\")
    console.print(f"  --cx {best['cx']} --cy {best['cy']} --size {window_size} --output \"best_match.png\"")

if __name__ == "__main__":
    app()
