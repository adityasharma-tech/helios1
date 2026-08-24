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
    jaxa_img: str = typer.Option(None, help="Path to JAXA PDS3 .img file"),
    jaxa_lbl: str = typer.Option(None, help="Path to JAXA PDS3 .lbl file"),
    query_img: str = typer.Option(None, help="Path to query square PNG image"),
    cx: int = typer.Option(..., help="Center X of ISRO region to extract"),
    cy: int = typer.Option(..., help="Center Y of ISRO region to extract"),
    size: int = typer.Option(5000, help="Size of ISRO region to extract (px)"),
    max_dim: int = typer.Option(1024, help="Max image dimension to prevent GPU OOM"),
    method: str = typer.Option("disk", help="Feature extraction method (disk or sift)"),
    output: str = typer.Option("result.png", help="Output visualization path"),
    plot_keypoints: bool = typer.Option(False, help="Plot all keypoints (red circles)")
):
    """Run the SIH Lunar Image Registration pipeline on real data.
    
    Uses either a JAXA image or a custom PNG as query (source) and searches inside the ISRO OHRC data (target).
    """
    if not os.path.exists(isro_img):
        console.print(f"[red]ISRO image not found: {isro_img}[/red]")
        return
        
    has_jaxa_context = False
    
    if query_img:
        console.print(f"[bold cyan]1. Reading Query Image (PNG)[/bold cyan]")
        source_raw = cv2.imread(query_img, cv2.IMREAD_GRAYSCALE)
        console.print(f"Source (PNG) Shape: {source_raw.shape}")
    else:
        if not jaxa_img or not jaxa_lbl:
            console.print("[red]Must provide either --query-img or both --jaxa-img and --jaxa-lbl[/red]")
            return
            
        # 1. Read JAXA as query/source
        console.print(f"[bold cyan]1. Reading JAXA Query Image[/bold cyan]")
        try:
            with JAXAReader(jaxa_img, jaxa_lbl) as reader:
                jaxa_full_h, jaxa_full_w = reader.height, reader.width
                console.print(f"JAXA Full Shape: {jaxa_full_h}x{jaxa_full_w}")
                source_raw = reader.read_all()
                
                # Extract full low-res JAXA for context (4th panel)
                jaxa_step = max(1, max(jaxa_full_h, jaxa_full_w) // 1000)
                jaxa_full_lowres = np.array(reader.mm[::jaxa_step, ::jaxa_step])
                has_jaxa_context = True
        except Exception as e:
            console.print(f"[red]Error reading JAXA data: {e}[/red]")
            return
            
        console.print(f"Source (JAXA) Shape: {source_raw.shape}")
    
    # 2. Read ISRO target region
    console.print(f"\n[bold cyan]2. Reading ISRO Target Region ({size}x{size}) at ({cx}, {cy})[/bold cyan]")
    try:
        with ISROReader(isro_img, isro_xml) as reader:
            isro_full_h, isro_full_w = reader.height, reader.width
            console.print(f"ISRO Full Shape: {isro_full_h}x{isro_full_w}")
            target_raw = reader.extract_patch(cx, cy, size)
            
            # Extract full low-res ISRO for context (1st panel)
            isro_step = max(1, max(isro_full_h, isro_full_w) // 1000)
            isro_full_lowres = np.array(reader.mm[::isro_step, ::isro_step])
            
            # Calculate ISRO box coordinates for the low-res context
            isro_box_x1 = int(max(0, cx - size//2) / isro_step)
            isro_box_y1 = int(max(0, cy - size//2) / isro_step)
            isro_box_x2 = int(min(isro_full_w, cx + size//2) / isro_step)
            isro_box_y2 = int(min(isro_full_h, cy + size//2) / isro_step)
            
    except Exception as e:
        console.print(f"[red]Error reading ISRO data: {e}[/red]")
        return
        
    console.print(f"Target (ISRO) Shape: {target_raw.shape}")
    
    console.print(f"\n[bold cyan]3. Radiometric Normalization (CLAHE)[/bold cyan]")
    source_img = normalize_16bit_to_8bit(source_raw)
    target_img = normalize_16bit_to_8bit(target_raw)
    console.print("Successfully scaled data to 8-bit representations.")
    
    console.print(f"\n[bold cyan]3.5. Anti-OOM Resizing[/bold cyan]")
    # Resize images if they exceed max_dim to prevent CUDA OOM
    def resize_for_gpu(img, max_dim):
        h, w = img.shape
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            console.print(f"Resizing {w}x{h} -> {new_w}x{new_h}")
            return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return img
        
    source_img = resize_for_gpu(source_img, max_dim)
    target_img = resize_for_gpu(target_img, max_dim)
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
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
        
        # ---- Visualization: Dynamic Panel Layout ----
        draw_flags = 0 if plot_keypoints else 2 # 2 = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        
        # drawMatches: img1=target (ISRO region), img2=source (query)
        match_vis = cv2.drawMatches(
            target_img, kp2, source_img, kp1,
            [cv2.DMatch(_queryIdx=m.trainIdx, _trainIdx=m.queryIdx, _distance=m.distance) 
             for i, m in enumerate(matches) if inliers[i]],
            None, matchColor=(0, 255, 0), singlePointColor=(0, 0, 255), flags=draw_flags
        )
        
        vis_h = match_vis.shape[0]
        
        # Panel 1: Full ISRO context with yellow box
        isro_ctx_8bit = normalize_16bit_to_8bit(isro_full_lowres)
        isro_scale = vis_h / isro_ctx_8bit.shape[0]
        isro_ctx_w = max(1, int(isro_ctx_8bit.shape[1] * isro_scale))
        isro_ctx = cv2.resize(isro_ctx_8bit, (isro_ctx_w, vis_h), interpolation=cv2.INTER_AREA)
        isro_ctx_bgr = cv2.cvtColor(isro_ctx, cv2.COLOR_GRAY2BGR)
        
        # Draw yellow box on ISRO context
        ib_x1 = int(isro_box_x1 * isro_scale)
        ib_y1 = int(isro_box_y1 * isro_scale)
        ib_x2 = int(isro_box_x2 * isro_scale)
        ib_y2 = int(isro_box_y2 * isro_scale)
        ib_thick = max(2, int(3 * isro_scale))
        cv2.rectangle(isro_ctx_bgr, (ib_x1, ib_y1), (ib_x2, ib_y2), (0, 255, 255), ib_thick)
        
        if has_jaxa_context:
            # Panel 4: Full JAXA context with yellow box
            jaxa_ctx_8bit = normalize_16bit_to_8bit(jaxa_full_lowres)
            jaxa_scale = vis_h / jaxa_ctx_8bit.shape[0]
            jaxa_ctx_w = max(1, int(jaxa_ctx_8bit.shape[1] * jaxa_scale))
            jaxa_ctx = cv2.resize(jaxa_ctx_8bit, (jaxa_ctx_w, vis_h), interpolation=cv2.INTER_AREA)
            jaxa_ctx_bgr = cv2.cvtColor(jaxa_ctx, cv2.COLOR_GRAY2BGR)
            
            jb_thick = max(2, int(3 * jaxa_scale))
            cv2.rectangle(jaxa_ctx_bgr, (0, 0), (jaxa_ctx_w - 1, vis_h - 1), (0, 255, 255), jb_thick)
            
            # Combine: [ISRO Context] | [Match Vis] | [JAXA Context]
            final_vis = np.hstack((isro_ctx_bgr, match_vis, jaxa_ctx_bgr))
        else:
            # Combine: [ISRO Context] | [Match Vis]
            final_vis = np.hstack((isro_ctx_bgr, match_vis))
        
        cv2.imwrite(output, final_vis)
        console.print(f"\n[green]Visualization saved to {output} with shape {final_vis.shape}[/green]")
    else:
        console.print("[red]Registration failed. Could not compute a valid geometric transformation (Not enough inliers).[/red]")

@app.command()
def train_gan(epochs: int = 2):
    """Train the GAN normalizer (Demo)."""
    from src.gan import train_gan_demo
    train_gan_demo(epochs=epochs)

if __name__ == "__main__":
    app()
