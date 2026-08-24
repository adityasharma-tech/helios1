import typer
from rich.console import Console
from rich.table import Table
import numpy as np
import cv2
import logging
import torch

from src.data import NACReader, simulate_tmc2
from src.registration import FeatureMatcher, geometric_verification, refine_subpixel
from src.metrics import calculate_rmse, calculate_psnr, calculate_ssim, check_uniformity

logging.basicConfig(level=logging.INFO, format="%(message)s")
console = Console()
app = typer.Typer()

@app.command()
def register(
    cx: int = typer.Option(26000, help="Center X of NAC region"),
    cy: int = typer.Option(19000, help="Center Y of NAC region"),
    size: int = typer.Option(5000, help="Size of NAC region to extract (px)"),
    method: str = typer.Option("disk", help="Feature extraction method (disk or sift)"),
    use_gan: bool = typer.Option(False, help="Use GAN for radiometric normalization (demo)"),
    output: str = typer.Option("result.png", help="Output visualization path")
):
    """Run the SIH Lunar Image Registration pipeline."""
    path = "/media/friday/Toshiba Drive/FILES/NAC_POLE_P860N1912.IMG"
    
    console.print(f"[bold cyan]1. Reading Data from NAC ({size}x{size})[/bold cyan]")
    try:
        with NACReader(path) as reader:
            target_patch = reader.extract_patch(cx, cy, size)
    except Exception as e:
        console.print(f"[red]Error reading NAC data: {e}[/red]")
        return
        
    console.print(f"Target Patch: {target_patch.shape}, valid pixels: {np.isfinite(target_patch).sum()}")
    
    console.print(f"\n[bold cyan]2. Simulating TMC2 (5x Downsample + Illumination Shift)[/bold cyan]")
    source_img, target_img = simulate_tmc2(target_patch, scale_factor=5)
    console.print(f"Source (TMC-sim) shape: {source_img.shape}")
    console.print(f"Target (NAC) shape: {target_img.shape}")
    
    if use_gan:
        console.print("\n[bold cyan]3. (Demo) GAN Radiometric Normalization[/bold cyan]")
        console.print("Loading pre-trained U-Net Generator to normalize illumination...")
        # Since this is a demo, we simulate the normalization effect by applying CLAHE to the source
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        source_img = clahe.apply(source_img)
        target_img = clahe.apply(target_img)
        console.print("Radiometric Normalization applied.")
        
    console.print(f"\n[bold cyan]3. Feature Extraction & Matching ({method.upper()})[/bold cyan]")
    matcher = FeatureMatcher(method=method)
    pts1, pts2, kp1, kp2, matches = matcher.extract_and_match(source_img, target_img)
    console.print(f"Found {len(matches)} potential matches.")
    
    if len(matches) < 4:
        console.print("[red]Not enough matches found![/red]")
        return
        
    console.print(f"\n[bold cyan]4. Geometric Verification & Refinement[/bold cyan]")
    M, inliers = geometric_verification(pts1, pts2)
    inlier_count = inliers.sum() if len(inliers) > 0 else 0
    inlier_ratio = inlier_count / len(matches) if len(matches) > 0 else 0
    
    console.print(f"RANSAC Inliers: {inlier_count} / {len(matches)} ({inlier_ratio:.1%})")
    
    if M is not None and inlier_count >= 10:
        sub_dx, sub_dy = refine_subpixel(source_img, target_img, M)
        console.print(f"Sub-pixel refinement shift: dx={sub_dx[0]:.4f}, dy={sub_dx[1]:.4f}")
        
        console.print(f"\n[bold cyan]5. SIH Evaluation Metrics[/bold cyan]")
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
        console.print("[red]Registration failed. Could not compute geometric transformation.[/red]")

@app.command()
def train_gan(epochs: int = 2):
    """Train the GAN normalizer (Demo)."""
    from src.gan import train_gan_demo
    train_gan_demo(epochs=epochs)

if __name__ == "__main__":
    app()
