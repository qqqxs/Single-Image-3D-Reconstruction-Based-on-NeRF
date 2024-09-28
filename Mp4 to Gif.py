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


clip = VideoFileClip("demo/fern.mp4")  # 需要转为GIF的视频文件路径
clip_gif = speedx(clip, factor=6)
# clip_gif = clip.speedx(2)
clip_gif.write_gif("demo/fern.gif", fps=60)

