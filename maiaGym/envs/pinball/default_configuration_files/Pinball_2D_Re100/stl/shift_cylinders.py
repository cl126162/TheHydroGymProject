import numpy as np

file = '/p/project/anpit/clagemann/maia_rl_environments/maia_rl_benchmark_Dec2024/maiaml/2D_pinball/stl/cylinder.stl'

data = np.loadtxt(file, skiprows=1, delimiter='	')

R = 0.5
shift_front = [-1.299, 0.0]
file_cylinder_front = '/p/project/anpit/clagemann/maia_rl_environments/maia_rl_benchmark_Dec2024/maiaml/2D_pinball/stl/cylinder_front.stl'
cylinder_front_data = data + shift_front
np.savetxt(file_cylinder_front, cylinder_front_data, header=str(len(data)), delimiter='	', fmt='%10.18f')

shift_up = [0.0, 0.75]
file_cylinder_up = '/p/project/anpit/clagemann/maia_rl_environments/maia_rl_benchmark_Dec2024/maiaml/2D_pinball/stl/cylinder_up.stl'
cylinder_up_data = data + shift_up
np.savetxt(file_cylinder_up, cylinder_up_data, header=str(len(data)), delimiter='	', fmt='%10.18f')


shift_down = [0.0, -0.75]
file_cylinder_down = '/p/project/anpit/clagemann/maia_rl_environments/maia_rl_benchmark_Dec2024/maiaml/2D_pinball/stl/cylinder_down.stl'
cylinder_down_data = data + shift_down
np.savetxt(file_cylinder_down, cylinder_down_data, header=str(len(data)), delimiter='	', fmt='%10.18f')


import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.figure()
plt.scatter(cylinder_front_data[:,0], cylinder_front_data[:,1])
plt.scatter(cylinder_up_data[:,0], cylinder_up_data[:,1])
plt.scatter(cylinder_down_data[:,0], cylinder_down_data[:,1])
plt.axis('equal')
plt.savefig('/p/project/anpit/clagemann/maia_rl_environments/maia_rl_benchmark_Dec2024/maiaml/2D_pinball/stl/pinball_arrangement.png')
