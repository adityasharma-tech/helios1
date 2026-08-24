import typer
from rich.console import Console
from rich.table import Table
import numpy as np
import cv2
import logging
import torch
import os

from src.data import ISROReader, JAXAReader, normalize_16bit_to_8bit
from src.registration import FeatureMatcher, geometric_verification, refine_subpixel
from src.metrics import calculate_rmse, calculate_psnr, calculate_ssim, check_uniformity

logging.basicConfig(level=logging.INFO, format="%(message)s")
console = Console()
app = typer.Typer()

@app.command()
def register(
    isro_img: str = typer.Option(..., help="Path to ISRO PDS4 .img file"),
    isro_xml: str = typer.Option(..., help="Path to ISRO PDS4 .xml file"),
    jaxa_img: str = typer.Option(..., help="Path to JAXA PDS3 .img file"),
    jaxa_lbl: str = typer.Option(..., help="Path to JAXA PDS3 .lbl file"),
    cx: int = typer.Option(..., help="Center X of ISRO region to extract"),
    cy: int = typer.Option(..., help="Center Y of ISRO region to extract"),
    size: int = typer.Option(5000, help="Size of ISRO region to extract (px)"),
    method: str = typer.Option("disk", help="Feature extraction method (disk or sift)"),
    output: str = typer.Option("result.png", help="Output visualization path")
):
    """Run the SIH Lunar Image Registration pipeline on real data."""
    if not os.path.exists(isro_img):
        console.print(f"[red]ISRO image not found: {isro_img}[/red]")
        return
        
    console.print(f"[bold cyan]1. Reading ISRO Source Image ({size}x{size}) at ({cx}, {cy})[/bold cyan]")
    try:
        with ISROReader(isro_img, isro_xml) as reader:
            console.print(f"ISRO Full Shape: {reader.height}x{reader.width}")
            source_raw = reader.extract_patch(cx, cy, size)
    except Exception as e:
        console.print(f"[red]Error reading ISRO data: {e}[/red]")
        return
        
    console.print(f"Source Patch Shape: {source_raw.shape}")
    
    console.print(f"\n[bold cyan]2. Reading JAXA Reference Image[/bold cyan]")
    try:
        with JAXAReader(jaxa_img, jaxa_lbl) as reader:
            console.print(f"JAXA Full Shape: {reader.height}x{reader.width}")
            # For this test, we read the entire JAXA image since it's only ~6000x5000
            target_raw = reader.read_all()
    except Exception as e:
        console.print(f"[red]Error reading JAXA data: {e}[/red]")
        return
        
    console.print(f"Target Shape: {target_raw.shape}")
    
    console.print(f"\n[bold cyan]3. Radiometric Normalization (CLAHE)[/bold cyan]")
    source_img = normalize_16bit_to_8bit(source_raw)
    target_img = normalize_16bit_to_8bit(target_raw)
    console.print("Successfully scaled 16-bit flat binaries to 8-bit representations.")
    
    console.print(f"\n[bold cyan]4. Feature Extraction & Matching ({method.upper()})[/bold cyan]")
    matcher = FeatureMatcher(method=method)
    pts1, pts2, kp1, kp2, matches = matcher.extract_and_match(source_img, target_img)
    console.print(f"Found {len(matches)} potential matches.")
    
    if len(matches) < 4:
        console.print("[red]Not enough matches found! Try a different region or method.[/red]")
        return
        
    console.print(f"\n[bold cyan]5. Geometric Verification & Refinement[/bold cyan]")
    M, inliers = geometric_verification(pts1, pts2)
    inlier_count = inliers.sum() if len(inliers) > 0 else 0
    inlier_ratio = inlier_count / len(matches) if len(matches) > 0 else 0
    
    console.print(f"RANSAC Inliers: {inlier_count} / {len(matches)} ({inlier_ratio:.1%})")
    
    if M is not None and inlier_count >= 5:
        # Wrap the refinement in try-catch as subpixel correlation can sometimes fail on large transformations
        try:
            sub_dx, sub_dy = refine_subpixel(source_img, target_img, M)
            console.print(f"Sub-pixel refinement shift: dx={sub_dx[0]:.4f}, dy={sub_dx[1]:.4f}")
        except:
            console.print("Sub-pixel refinement skipped.")
            
        console.print(f"\n[bold cyan]6. SIH Evaluation Metrics[/bold cyan]")
        rmse = calculate_rmse(pts1, pts2, M, inliers)
        
        # Warp source to target space to compare images
        h, w = target_img.shape
        warped_source = cv2.warpPerspective(source_img, M, (w, h))
        psnr = calculate_psnr(warped_source, target_img)
        ssim = calculate_ssim(warped_source, target_img)
        
        uniformity = check_uniformity(pts1[inliers], source_img.shape)
        
        table = Table(box=None)
        table.add_column("Metric", style="bold")
        table.add_column("Value")
        table.add_row("RMSE (Accuracy)", f"{rmse:.2f} px")
        table.add_row("PSNR (Radiometric)", f"{psnr:.2f} dB")
        table.add_row("SSIM (Structural)", f"{ssim:.4f}")
        table.add_row("Inlier Ratio", f"{inlier_ratio:.1%}")
        table.add_row("Match Uniformity", f"{uniformity:.1%} covered")
        console.print(table)
        
        # Visualization
        match_vis = cv2.drawMatches(
            source_img, kp1, target_img, kp2,
            [m for i, m in enumerate(matches) if inliers[i]],
            None, matchColor=(0, 255, 0), singlePointColor=(0, 0, 255), flags=0
        )
        cv2.imwrite(output, match_vis)
        console.print(f"\n[green]Visualization saved to {output}[/green]")
    else:
        console.print("[red]Registration failed. Could not compute a valid geometric transformation (Not enough inliers).[/red]")

@app.command()
def train_gan(epochs: int = 2):
    """Train the GAN normalizer (Demo)."""
    from src.gan import train_gan_demo
    train_gan_demo(epochs=epochs)

if __name__ == "__main__":
    app()
