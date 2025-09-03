import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import imageio


output_filename = '../output_CFL1.txt'
data = np.loadtxt(output_filename)

x = data[0, 1:]
u = data[1:, 1:]
t = data[1:, 0]

def fplot(i):
    fig,ax = plt.subplots(figsize=(8,6))
    ax.plot(x, u[i])
    ax.set_title(f'Temps = {t[i]:.2f} secondes')
    ax.set_xlabel('Position (x)')
    ax.set_ylabel('Amplitude (u)')
    ax.yaxis.set_label_coords(-0.1, 0.5)
    ax.set_xlim(np.min(x), np.max(x))
    ax.set_ylim(np.min(u[i]), 1.1*np.max(u[i]))
    ax.grid()
    return fig

frames = []
for i in range(len(t)):
    fig = fplot(i)
    fig.canvas.draw()
    frames.append(np.array(fig.canvas.renderer._renderer))
    plt.close("all")

gif_filename = output_filename.split('.')[-2].split("/")[-1] + '.gif'
imageio.mimsave(gif_filename, frames, fps=20)
print(f'GIF saved as {gif_filename}')

