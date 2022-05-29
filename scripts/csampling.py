import sys

parameter = sys.argv[1]
coherence = sys.argv[2]

from pathlib import Path
import time
import numpy as np
import wwparrecdl.cfast.parrec as cpr

parent_path = Path(__file__).parent.parent.absolute()
data_dir = parent_path / 'wwparrecdl/data'

start = time.time()
outcomes = cpr.wangwong()
end = time.time()

print("Generated {} samples of {} trials in {} min.".format(10000, 10000, (end-start)/60))

np.save(str(data_dir / parameter / coherence / "times/times.npy"), outcomes[0])
np.save(str(data_dir / parameter / coherence / "choices/choices.npy"), outcomes[1])
np.save(str(data_dir / parameter / coherence / "parameters/parameters.npy"), outcomes[2])
