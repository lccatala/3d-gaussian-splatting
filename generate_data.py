import os
import argparse
import math

import numpy as np
import mathutils
import bpy


def _save_matrix(matrix, filepath: str) -> None:
    np.save(filepath, matrix)
    print(f"Saved\n{matrix}")

def generate_images_and_matrices(scene_path: str, num_cameras: int, output_dir: str):
    bpy.ops.wm.open_mainfile(filepath=scene_path) #type: ignore
    scene = bpy.context.scene #type: ignore
    center = mathutils.Vector((0, 0, 0))
    radius = 10 # Distance of cameras from center
    angle_step = 360 / num_cameras
    for i in range(num_cameras):
        angle = math.radians(i * angle_step)
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        z = radius / 2

        # Create a new camera
        bpy.ops.object.camera_add(location=(x, y, z)) #type: ignore
        camera = bpy.context.object #type: ignore

        # Point the camera towards the center
        direction = center - mathutils.Vector(camera.location)
        rot_quat = direction.to_track_quat("-Z", "Y")
        camera.rotation_euler = rot_quat.to_euler()

        # Set the camera as active
        scene.camera = camera

        # Render the image
        image_path = os.path.join(output_dir, f"camera_{i:03d}.png")
        scene.render.filepath = image_path
        bpy.ops.render.render(write_still=True) #type: ignore

        # Save the camera view matrix
        view_matrix = camera.matrix_world.inverted()
        view_matrix_path = os.path.join(output_dir, f"camera_{i:03d}_view_matrix.npy")
        _save_matrix(view_matrix, view_matrix_path)

        # Save the camera projection matrix
        projection_matrix = camera.calc_matrix_camera(bpy.context.view_layer.depsgraph) #type: ignore
        projection_matrix_path = os.path.join(output_dir, f"camera_{i:03d}_projection_matrix.npy")
        _save_matrix(projection_matrix, projection_matrix_path)

        bpy.data.objects.remove(camera, do_unlink=True) #type: ignore

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""
        Extract renders and their corresponding projection and view matrices for the training parts of the project, from multiple camera angles around the center of a given Blender scene.""")
    parser.add_argument("--scene_path", type=str, help="Path to the .blend file.")
    parser.add_argument("--output_dir", type=str, help="Directory to save rendered images and camera matrices.")
    parser.add_argument("--num_cameras", type=int, default=10, help="Number of cameras to create.")
    parser.add_argument("--radius", type=float, default=10.0, help="Radius of the camera placement.")
    args = parser.parse_args()

    generate_images_and_matrices(args.scene_path, args.num_cameras, args.output_dir)

