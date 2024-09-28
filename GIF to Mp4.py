import moviepy.editor as mp


vfc = mp.VideoFileClip("demo/house-rgb.gif")
vfc.write_videofile("demo/house-rgb.mp4")
