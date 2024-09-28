from moviepy.editor import *
from moviepy.editor import VideoFileClip, vfx, clips_array


def speedx(clip, factor=None, final_duration=None):
    """
    Returns a clip playing the current clip but at a speed multiplied
    by ``factor``. Instead of factor one can indicate the desired
    ``final_duration`` of the clip, and the factor will be automatically
    computed.
    The same effect is applied to the clip's audio and mask if any.
    """

    if final_duration:
        factor = 1.0 * clip.duration / final_duration

    newclip = clip.fl_time(lambda t: factor * t, apply_to=['mask', 'audio'])

    if clip.duration is not None:
        newclip = newclip.set_duration(1.0 * clip.duration / factor)

    return newclip


def time_mirror(self):
    """
    Returns a clip that plays the current clip backwards.
    The clip must have its ``duration`` attribute set.
    The same effect is applied to the clip's audio and mask if any.
    """
    return self.fl_time(lambda t: self.duration - t, keep_duration=True)


clip = VideoFileClip("demo/chimpanzee-grin-normal.mp4")  # 需要转为GIF的视频文件路径
clip_reverse = clip.fx(time_mirror)

# 将两个剪辑连接起来
final_clip = concatenate_videoclips([clip, clip_reverse])
final_clip = speedx(final_clip, factor=12)

# 将连接后的剪辑转换为GIF
final_clip.write_gif("demo/chimpanzee-grin-normal.gif", fps=60)
