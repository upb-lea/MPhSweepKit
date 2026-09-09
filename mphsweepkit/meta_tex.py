from matplotlib import pyplot as plt

# Matplotlib settings
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "Bitstream Vera Sans",
        "font.size": 9.0,
        "text.latex.preamble": r"\usepackage{upgreek}\usepackage{siunitx}",
        "mathtext.fontset": "custom",
        "mathtext.rm": "Bitstream Vera Serif",
        "mathtext.it": "Bitstream Vera Serif:italic",
        "mathtext.bf": "Bitstream Vera Serif:bold",
    }
)
