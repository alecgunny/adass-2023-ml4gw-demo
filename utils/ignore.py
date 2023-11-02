"""
Ignore a bunch of ugly warnings that muddy
up our otherwise beautiful notebooks. In
practice, never ever do this.
"""

import warnings

warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")

from gwpy.io.nds2 import NDSWarning  # noqa

# for the data notebook
warnings.filterwarnings("ignore", module="gwpy", category=NDSWarning)
warnings.filterwarnings("ignore", module="pandas", category=UserWarning)
warnings.filterwarnings("ignore", module="tqdm")

# for the main notebook
warnings.filterwarnings("ignore", module="gwpy")
warnings.filterwarnings("ignore", module="lightning")
