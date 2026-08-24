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
    """Run the SIH Lunar Image Registration pipeline on real data."""
    if not os.path.exists(isro_img):
        console.print(f"[red]ISRO image not found: {isro_img}[/red]")
        return
        
    if query_img:
        console.print(f"[bold cyan]1. Reading Query Image (Square)[/bold cyan]")
        source_img = cv2.imread(query_img, cv2.IMREAD_GRAYSCALE)
        console.print(f"Query Shape: {source_img.shape}")
        
        console.print(f"\n[bold cyan]2. Reading Target Region from ISRO Large Data ({size}x{size}) at ({cx}, {cy})[/bold cyan]")
        with ISROReader(isro_img, isro_xml) as reader:
            console.print(f"ISRO Full Shape: {reader.height}x{reader.width}")
            target_raw = reader.extract_patch(cx, cy, size)
            
            # Extract full low-res image for context
            step_y = max(1, reader.height // 1000)
            step_x = max(1, reader.width // 1000)
            step = min(step_y, step_x) # keep aspect ratio
            full_lowres = np.array(reader.mm[::step, ::step])
            
            # Calculate box coordinates for the low-res image
            box_x1 = int(max(0, cx - size//2) / step)
            box_y1 = int(max(0, cy - size//2) / step)
            box_x2 = int(min(reader.width, cx + size//2) / step)
            box_y2 = int(min(reader.height, cy + size//2) / step)
            
        target_img = normalize_16bit_to_8bit(target_raw)
        console.print(f"Target Shape: {target_img.shape}")
        
    else:
        console.print(f"[bold cyan]1. Reading ISRO Source Image ({size}x{size}) at ({cx}, {cy})[/bold cyan]")
        try:
            with ISROReader(isro_img, isro_xml) as reader:
                console.print(f"ISRO Full Shape: {reader.height}x{reader.width}")
                source_raw = reader.extract_patch(cx, cy, size)
                
                # Extract full low-res image for context
                step_y = max(1, reader.height // 1000)
                step_x = max(1, reader.width // 1000)
                step = min(step_y, step_x) # keep aspect ratio
                full_lowres = np.array(reader.mm[::step, ::step])
                
                # Calculate box coordinates for the low-res image
                box_x1 = int(max(0, cx - size//2) / step)
                box_y1 = int(max(0, cy - size//2) / step)
                box_x2 = int(min(reader.width, cx + size//2) / step)
                box_y2 = int(min(reader.height, cy + size//2) / step)
                
        except Exception as e:
            console.print(f"[red]Error reading ISRO data: {e}[/red]")
            return
            
        console.print(f"Source Patch Shape: {source_raw.shape}")
        
        console.print(f"\n[bold cyan]2. Reading JAXA Reference Image[/bold cyan]")
        try:
            with JAXAReader(jaxa_img, jaxa_lbl) as reader:
                console.print(f"JAXA Full Shape: {reader.height}x{reader.width}")
                target_raw = reader.read_all()
        except Exception as e:
            console.print(f"[red]Error reading JAXA data: {e}[/red]")
            return
            
        console.print(f"Target Shape: {target_raw.shape}")
        
        console.print(f"\n[bold cyan]3. Radiometric Normalization (CLAHE)[/bold cyan]")
        source_img = normalize_16bit_to_8bit(source_raw)
        target_img = normalize_16bit_to_8bit(target_raw)
        console.print("Successfully scaled 16-bit flat binaries to 8-bit representations.")
    
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
        
        # Visualization
        draw_flags = 0 if plot_keypoints else 2 # 2 = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        
        # We want: [Context] | [Target (What we found)] | [Source (Image we are searching)]
        # cv2.drawMatches puts img1 on left, img2 on right. 
        # By passing target_img as img1 and source_img as img2, target is middle, source is right.
        swapped_matches = []
        for i, m in enumerate(matches):
            if inliers[i]:
                # Swap queryIdx and trainIdx
                swapped_matches.append(cv2.DMatch(_queryIdx=m.trainIdx, _trainIdx=m.queryIdx, _distance=m.distance))
                
        match_vis = cv2.drawMatches(
            target_img, kp2, source_img, kp1,
            swapped_matches,
            None, matchColor=(0, 255, 0), singlePointColor=(0, 0, 255), flags=draw_flags
        )
        
        # Prepare context visualization
        full_8bit = normalize_16bit_to_8bit(full_lowres)
        
        # Resize context image to match the height of match_vis
        target_h = match_vis.shape[0]
        scale = target_h / full_8bit.shape[0]
        new_w = max(1, int(full_8bit.shape[1] * scale))
        context_img_resized = cv2.resize(full_8bit, (new_w, target_h), interpolation=cv2.INTER_AREA)
        context_bgr = cv2.cvtColor(context_img_resized, cv2.COLOR_GRAY2BGR)
        
        # Draw bounding box on resized context image
        r_box_x1 = int(box_x1 * scale)
        r_box_y1 = int(box_y1 * scale)
        r_box_x2 = int(box_x2 * scale)
        r_box_y2 = int(box_y2 * scale)
        
        # Ensure box is visible even if small
        box_thickness = max(2, int(3 * scale))
        cv2.rectangle(context_bgr, (r_box_x1, r_box_y1), (r_box_x2, r_box_y2), (0, 255, 255), box_thickness)
        
        # Combine context image with match visualization
        final_vis = np.hstack((context_bgr, match_vis))
        
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
