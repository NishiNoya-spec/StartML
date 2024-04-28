import matplotlib.pyplot as plt
import matplotlib as mlp
import warnings

def set_visualization_settings():

    plt.style.use('dark_background')

    mlp.rcParams['lines.linewidth'] = 2

    mlp.rcParams['xtick.major.size'] = 12
    mlp.rcParams['xtick.major.width'] = 2
    mlp.rcParams['xtick.labelsize'] = 10
    mlp.rcParams['xtick.color'] = '#FF5533'

    mlp.rcParams['ytick.major.size'] = 12
    mlp.rcParams['ytick.major.width'] = 2
    mlp.rcParams['ytick.labelsize'] = 10
    mlp.rcParams['ytick.color'] = '#FF5533'

    mlp.rcParams['axes.labelsize'] = 10
    mlp.rcParams['axes.titlesize'] = 16
    mlp.rcParams['axes.titlecolor'] = '#00B050'
    mlp.rcParams['axes.labelcolor'] = '#00B050'

    warnings.filterwarnings('ignore')

