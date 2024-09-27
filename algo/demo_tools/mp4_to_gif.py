'''
Author: Qing Hong
Date: 2024-04-15 17:01:30
LastEditors: QingHong
LastEditTime: 2024-04-15 17:11:28
Description: file content
'''
from moviepy.editor import VideoFileClip

def convert_mp4_to_gif(source_path, target_path, fps=10):
    """
    Convert an MP4 video to a GIF.

    Parameters:
        source_path (str): The path to the source MP4 file.
        target_path (str): The path to save the GIF file.
        fps (int): Frames per second (default 10) for the output GIF.
    """
    # Load the source video file
    clip = VideoFileClip(source_path)
    
    # Write the GIF file
    clip.write_gif(target_path, fps=fps)

# Usage
root = '/Users/qhong/Downloads/sample_1.mp4'
convert_mp4_to_gif(root, root.replace('.mp4','.gif'), fps=10)