import os
import argparse
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import cv2


DEVICE = "cpu"

class GaussianSplat:
    def __init__(
        self, 
        position: torch.Tensor, 
        covariance: torch.Tensor, 
        color: torch.Tensor, 
        opacity: torch.Tensor, 
        sh_coeffs: Optional[torch.Tensor]=None
    ):
        """
        Initialize a single Gaussian splat
        
        Args:
            position (torch.Tensor): 3D position (x,y,z)
            covariance (torch.Tensor): 3x3 covariance matrix
            color (torch.Tensor): RGB color values
            opacity (torch.Tensor): Alpha/opacity value
            sh_coeffs (torch.Tensor, optional): Spherical harmonics coefficients
        """

        self.position = nn.Parameter(position)
        self.covariance = nn.Parameter(covariance)
        self.color = nn.Parameter(color)
        self.opacity = nn.Parameter(opacity)
        self.sh_coeffs = nn.Parameter(sh_coeffs) if sh_coeffs is not None else None


class GaussianSplattingModel:
    def __init__(self, num_gaussians):
        """
        Initialize the 3D Gaussian Splatting model
        
        Args:
            num_gaussians (int): Number of Gaussian splats to initialize
        """
        self.gaussians = []
        self.initialize_gaussians(num_gaussians)
        
    def initialize_gaussians(self, num_gaussians):
        """Initialize Gaussian splats with random parameters"""
        for _ in range(num_gaussians):
            position = torch.randn(3, device=DEVICE)
            # Initialize as isotropic Gaussians
            covariance = torch.eye(3, device=DEVICE)
            color = torch.rand(3, device=DEVICE)
            opacity = torch.rand(1, device=DEVICE)
            
            splat = GaussianSplat(position, covariance, color, opacity)
            self.gaussians.append(splat)
    
    def project_gaussian_to_2d(self, gaussian, camera_matrix):
        """
        Project 3D Gaussian to 2D screen space
        
        Args:
            gaussian (GaussianSplat): Gaussian to project
            camera_matrix (torch.Tensor): 4x4 camera projection matrix
        
        Returns:
            Tuple containing 2D mean and 2D covariance matrix
        """
        # Project 3D position to 2D
        pos_homogeneous = torch.cat([gaussian.position, torch.ones(1, device=DEVICE)])
        pos_2d = camera_matrix.float() @ pos_homogeneous.float()
        pos_2d = pos_2d[:2] / pos_2d[2]  # Perspective division
        
        # Project covariance to 2D (simplified)
        J = camera_matrix[:2, :3]  # Jacobian of the projection
        cov_2d = J.float() @ gaussian.covariance.float() @ J.T.float()
        
        return pos_2d, cov_2d
    
    def compute_2d_gaussian(self, mean_2d, cov_2d, image_size):
        """
        Compute 2D Gaussian parameters for rendering
        
        Args:
            mean_2d (torch.Tensor): 2D projected position
            cov_2d (torch.Tensor): 2D covariance matrix
            image_size (tuple): Target image dimensions
        
        Returns:
            torch.Tensor: 2D Gaussian evaluation on pixel grid
        """
        x = torch.arange(image_size[1], device=DEVICE)
        y = torch.arange(image_size[0], device=DEVICE)
        Y, X = torch.meshgrid(y, x, indexing="ij")
        pixels = torch.stack([X, Y], dim=-1)
        
        # Compute Gaussian values for each pixel
        diff = pixels - mean_2d.unsqueeze(0).unsqueeze(0)
        inv_cov = torch.inverse(cov_2d + torch.eye(2, device=DEVICE) * 1e-5)
        mahalanobis = torch.sum((diff @ inv_cov) * diff, dim=-1)
        gaussian = torch.exp(-0.5 * mahalanobis)
        
        return gaussian

    def render(self, camera_matrix, image_size):
        """
        Render the scene from a given camera viewpoint
        
        Args:
            camera_matrix (torch.Tensor): 4x4 camera projection matrix
            image_size (tuple): Target image dimensions (H, W)
            
        Returns:
            torch.Tensor: Rendered image
        """
        rendered_image = torch.zeros(*image_size, 3, device=DEVICE)
        accumulated_alpha = torch.zeros(*image_size, device=DEVICE)
        
        # Sort Gaussians by depth (back-to-front)
        depths = [
            (g, (camera_matrix.float() @ torch.cat([g.position.float(), torch.ones(1, device=DEVICE)]))[2])
            for g in self.gaussians
        ]
        sorted_gaussians = [g for g, _ in sorted(depths, key=lambda x: x[1])]
        
        for gaussian in sorted_gaussians:
            # Project Gaussian to 2D
            mean_2d, cov_2d = self.project_gaussian_to_2d(gaussian, camera_matrix)
            
            # Skip if outside view frustum
            if not (0 <= mean_2d[0] < image_size[1] and 0 <= mean_2d[1] < image_size[0]):
                continue
            
            # Compute 2D Gaussian footprint
            footprint = self.compute_2d_gaussian(mean_2d, cov_2d, image_size)
            
            # Alpha compositing
            alpha = footprint * gaussian.opacity 
            alpha_expanded = alpha.unsqueeze(-1) 
            color = gaussian.color.view(1, 1, 3).expand(image_size[0], image_size[1], 3)
            color_contribution = color * alpha_expanded
            
            rendered_image = rendered_image * (1 - alpha).unsqueeze(-1) + color_contribution
            accumulated_alpha = accumulated_alpha * (1 - alpha) + alpha
            
        return rendered_image

def create_loss_function():
    """
    Create loss function for optimization
    """
    def loss_fn(rendered_image, target_image) -> torch.Tensor:
        # L2 loss between rendered and target images
        mse_loss = torch.mean((rendered_image - target_image) ** 2)
        return mse_loss
    return loss_fn

def optimize_gaussians(
    model: GaussianSplattingModel, 
    target_images: list[torch.Tensor], 
    camera_matrices: list[np.ndarray], 
    num_iterations: int = 1000):
    """
    Optimize Gaussian parameters to match target images
    
    Args:
        model (GaussianSplatting): The model to optimize
        target_images (list): List of target images
        camera_matrices (list): List of camera matrices corresponding to target images
        num_iterations (int): Number of optimization iterations
    """
    optimizer = torch.optim.Adam([p for g in model.gaussians for p in [g.position, g.covariance, g.color, g.opacity]], lr=0.01)
    loss_fn = create_loss_function()
    
    for iteration in range(num_iterations):
        optimizer.zero_grad()
        total_loss: torch.Tensor = torch.tensor(0.0, requires_grad=True, device=DEVICE)
        
        # Compute loss for each view
        for target_image, camera_matrix in zip(target_images, camera_matrices):
            rendered_image = model.render(camera_matrix, target_image.shape[:2])
            loss = loss_fn(rendered_image, target_image)
            total_loss = total_loss + loss
        
        total_loss.backward()
        optimizer.step()
        
        if iteration % 100 == 0:
            print(f"Iteration {iteration}, Loss: {total_loss.item():.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D Gaussian Splatting main script")
    parser.add_argument("--input_dir", type=str, help="Directory with rendered images and camera matrices.")
    parser.add_argument("--num_gaussians", type=int, help="Number of Gaussians", default=1000)
    args = parser.parse_args()

    if torch.cuda.is_available():
        DEVICE = "cuda"
    elif torch.backends.mps.is_available():
        DEVICE = "mps"

    model = GaussianSplattingModel(num_gaussians=args.num_gaussians)

    target_images = []
    camera_matrices = []
    data_files = os.listdir(args.input_dir)
    image_files: list[str] = [f for f in data_files if f.endswith(".png")]
    for image_file in image_files:
        projection_matrix_file = f"{image_file.split('.')[0]}_projection_matrix.npy"
        pm_filepath = os.path.join(args.input_dir, projection_matrix_file)
        image_filepath = os.path.join(args.input_dir, image_file)

        matrix = torch.from_numpy(np.load(pm_filepath)).to(DEVICE, dtype=torch.float32)
        camera_matrices.append(matrix)

        image = torch.from_numpy(cv2.imread(image_filepath)).to(DEVICE, dtype=torch.float32)
        target_images.append(image)

    optimize_gaussians(model, target_images, camera_matrices)
