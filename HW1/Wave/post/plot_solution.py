import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import imageio
plt.rcParams.update({'font.size': 14})

# output_filename = '../output_CFL1.txt'
# data = np.loadtxt(output_filename)

# x = data[0, 1:]
# u = data[1:, 1:]
# t = data[1:, 0]

# def fplot(i):
#     fig,ax = plt.subplots(figsize=(8,6))
#     ax.plot(x, u[i])
#     ax.set_title(f'Temps = {t[i]:.2f} secondes')
#     ax.set_xlabel('Position (x)')
#     ax.set_ylabel('Amplitude (u)')
#     ax.yaxis.set_label_coords(-0.1, 0.5)
#     ax.set_xlim(np.min(x), np.max(x))
#     ax.set_ylim(min(1.1*np.min(u[i]),0), max(1.1*np.max(u[i]),0))
#     ax.grid()
#     return fig

# frames = []
# for i in range(len(t)):
#     fig = fplot(i)
#     fig.canvas.draw()
#     frames.append(np.array(fig.canvas.renderer._renderer))
#     plt.close("all")

# gif_filename = output_filename.split('.')[-2].split("/")[-1] + '.gif'
# imageio.mimsave('output/' + method.lower() + '.gif', frames, fps=20)
# # imageio.mimsave('output/' + gif_filename, frames, fps=20)
# print(f'GIF saved as {method.lower()}.gif')




# output_filename_obs = '../output_CFL1_obs.txt'
# data_obs = np.loadtxt(output_filename_obs)
# method = 'Hybride_Theta=1.0'

# u_obs = data_obs[1, 1:]
# t_obs = data_obs[1, 0]

# plt.figure()
# plt.plot(x, u_obs)
# plt.title(f'{method.replace("_"," ")} \n Temps = {t_obs:.2f} secondes')
# plt.xlabel('Position (x)')
# plt.ylabel('Amplitude (u)')
# plt.gca().yaxis.set_label_coords(-0.1, 0.5)
# plt.xlim(np.min(x), np.max(x))
# plt.ylim(min(1.1*np.min(u_obs),0), max(1.1*np.max(u_obs),0))
# plt.grid()
# plt.tight_layout()
# plt.savefig('output/' + method.lower() + '.pdf')

algorithms = ["explicitBackward", "explicitForward", "forwardTimeCenteredSpace", "leapFrog",
              "laxWendroff", "lax", "hybridExplicitImplicit_theta0", "hybridExplicitImplicit_theta0-5",
              "hybridExplicitImplicit_theta1", "tremblayTran"]

CFL = {"explicitBackward" : ["0-5", "0-75", "1"],
       "explicitForward" : ["0-5", "0-75", "1"],
       "forwardTimeCenteredSpace" : ["0-25", "0-5", "1"],
       "leapFrog" : ["0-25", "0-5", "1"],
       "laxWendroff" : ["0-25", "0-5", "1"],
       "lax" : ["0-25", "0-5", "1"],
       "hybridExplicitImplicit_theta0" : ["0-5", "1", "1-5", "2"],
       "hybridExplicitImplicit_theta0-5" : ["0-5", "1", "1-5", "2"],
       "hybridExplicitImplicit_theta1" : ["0-5", "1", "1-5", "2"],
       "tremblayTran" : ["0-5", "1", "1-5", "1-7"]}

for method in algorithms:
    plt.figure()
    for cfl in CFL[method]:
        output_filename = f'../output/{method}_CFL{cfl}_output_obs.txt'
        data = np.loadtxt(output_filename)
        t_obs = data[1, 0]
        x = data[0, 1:]
        u = data[1, 1:]
        plt.plot(x, u, label=f'CFL = {cfl.replace("-",".")}')
    plt.title(f'{method.replace("_"," ")} at t = {t_obs:.2f} seconds')
    plt.xlabel('Position (x)')
    plt.ylabel('Amplitude (u)')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(method.lower() + '.pdf', format='pdf')