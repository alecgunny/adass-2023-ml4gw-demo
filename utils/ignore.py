"""
Ignore a bunch of ugly warnings that muddy
up our otherwise beautiful notebooks. In
practice, never ever do this.
"""

import logging
import warnings

# need to ignore lal warnings before we attempt
# a gwpy import, otherwise the warning will get
# triggered by gwpy's internal import of lal
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")

from gwpy.io.nds2 import NDSWarning  # noqa

# for the data notebook
warnings.filterwarnings("ignore", module="gwpy", category=NDSWarning)
warnings.filterwarnings("ignore", module="pandas", category=UserWarning)
warnings.filterwarnings("ignore", module="tqdm")
warnings.filterwarnings("ignore", module="torch")

# for the main notebook
logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(
    logging.WARNING
)  # noqa
logging.getLogger("lightning.pytorch.accelerators.cuda").setLevel(
    logging.WARNING
)  # noqa
warnings.filterwarnings("ignore", module="gwpy")
warnings.filterwarnings("ignore", module="lightning")
