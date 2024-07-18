'''
Author: Qing Hong
FirstEditTime: This function has been here since 1987. DON'T FXXKING TOUCH IT
LastEditors: Qing Hong
LastEditTime: 2024-06-19 10:42:38
Description: 
         ▄              ▄
        ▌▒█           ▄▀▒▌     
        ▌▒▒▀▄       ▄▀▒▒▒▐
       ▐▄▀▒▒▀▀▀▀▄▄▄▀▒▒▒▒▒▐     ,-----------------.
     ▄▄▀▒▒▒▒▒▒▒▒▒▒▒█▒▒▄█▒▐     (Wow,kousei's code)
   ▄▀▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▀██▀▒▐     `-,---------------' 
  ▐▒▒▒▄▄▄▒▒▒▒▒▒▒▒▒▒▒▒▒▀▄▒▒▌  _.-'   ,----------.
  ▌▒▒▐▄█▀▒▒▒▒▄▀█▄▒▒▒▒▒▒▒█▒▐         (surabashii)
 ▐▒▒▒▒▒▒▒▒▒▒▒▀██▀▒▒▒▒▒▒▒▒▀▄▌        `-,--------' 
 ▌▒▀▄██▄▒▒▒▒▒▒▒▒▒▒▒░░░░▒▒▒▒▌      _.-'
 ▌▀▐▄█▄█▌▄▒▀▒▒▒▒▒▒░░░░░░▒▒▒▐ _.-'
▐▒▀▐▀▐▀▒▒▄▄▒▄▒▒▒▒▒░░░░░░▒▒▒▒▌
▐▒▒▒▀▀▄▄▒▒▒▄▒▒▒▒▒▒░░░░░░▒▒▒▐
 ▌▒▒▒▒▒▒▀▀▀▒▒▒▒▒▒▒▒░░░░▒▒▒▒▌
 ▐▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▐
  ▀▄▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▄▒▒▒▒▌
    ▀▄▒▒▒▒▒▒▒▒▒▒▄▄▄▀▒▒▒▒▄▀
      ▀▄▄▄▄▄▄▀▀▀▒▒▒▒▒▄▄▀
         ▒▒▒▒▒▒▒▒▒▒▀▀
When I wrote this, only God and I understood what I was doing
Now, God only knows
'''
# 打开文件并读取内容到缓冲区
import ffmpeg
import imageio
vi = '/Users/qhong/Downloads/638BA148-1144-4762-B4CC-B1162146C55E.MOV'
# 打开视频文件
video = imageio.get_reader(vi, 'ffmpeg')

import ffmpeg

# 假设深度信息存储在视频文件中的第二个流中
(
    ffmpeg.input(vi).output('/Users/qhong/Documents/1117test/MM/motionmodel/algo/conversion_tools/pointcloud/depth_output.raw', map=2).run()
)
# ffmpeg.probe()

probe = ffmpeg.probe(vi)
video_info = next(stream for stream in probe['streams'] if stream['codec_type'] == 'video')
depth = next((stream for stream in probe['streams'] if stream['codec_type'] == 'data'), None)

stream = ffmpeg.input(vi).filter('select', f'index=2')
stream = ffmpeg.output(stream, '/Users/qhong/Documents/1117test/MM/motionmodel/algo/conversion_tools/pointcloud/output.mov')
ffmpeg.run(stream)

stream = ffmpeg.input(vi, stream_index=depth['index'])
stream = ffmpeg.output(stream, '/Users/qhong/Documents/1117test/MM/motionmodel/algo/conversion_tools/pointcloud/output.mov')
ffmpeg.run(stream)
# 读取生成的深度信息文件
with open('depth_output.raw', 'rb') as f:
    depth_data = f.read()
    # 处理深度数据
    print(depth_data)



from moviepy.editor import VideoFileClip


# 加载视频文件
video = VideoFileClip(vi)

# 获取指定索引的视频流
stream_index = 2
video_stream = video.subclip(stream_index, stream_index+1)

# 保存视频流为新文件
video_stream.write_videofile(output_file, codec='libx264', audio_codec='aac')




import av
import numpy as np

def read_depth_from_mov(file_path, stream_index=2):
    container = av.open(file_path)
    
    # 获取深度流
    stream = container.streams[stream_index]

    depth_frames = []
    # for packet in container.demux(stream):
    #     for frame in packet.decode():
    #         depth_frames.append(frame.to_ndarray())
    for packet in container.demux(stream):
        for frame in packet:
            # 将帧转换为NumPy数组
            frame_array = frame.to_ndarray()
            depth_frames.append(frame_array)
    # 将所有帧堆叠成一个NumPy数组
    depth_array = np.stack(depth_frames)
    return depth_array

# 使用示例
file_path = '/Users/qhong/Downloads/638BA148-1144-4762-B4CC-B1162146C55E.MOV'
depth_array = read_depth_from_mov(file_path,stream_index=1)
print(depth_array.shape)