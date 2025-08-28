import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

output_filename = '../output_CFL05.txt'
data = np.loadtxt(output_filename)

x = data[0, 1:]
u = data[1:, 1:]
t = data[1:, 0]

# If u is 1D (single time step), make it 2D for consistency
if u.ndim == 1:
    u = u[np.newaxis, :]

fig, ax = plt.subplots()
line, = ax.plot([], [], lw=2)
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, bbox=dict(facecolor='white', edgecolor='none', alpha=1.0))
ax.set_xlim(np.min(x), np.max(x))
ax.set_ylim(np.min(u), 1.1*np.max(u))
ax.set_xlabel('x')
ax.set_ylabel('u')
ax.set_title('Wave Solution Evolution')
ax.grid(True)

def init():
    line.set_data([], [])
    time_text.set_text('')
    return line, time_text

def animate(i):
    line.set_data(x, u[i])
    time_text.set_text(f'time = {t[i]:.3f}')
    return line, time_text

ani = animation.FuncAnimation(fig, animate, frames=u.shape[0], init_func=init,
                              blit=True, interval=50)

gif_filename = output_filename.split('.')[-2].split("/")[-1] + '.gif'
ani.save(gif_filename, writer='pillow')
print(f'GIF saved as {gif_filename}')