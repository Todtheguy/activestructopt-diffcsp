from activestructopt.sampler.diffusion import Diffusion
from activestructopt.common.materialsproject import get_structure
import os

api_key = os.getenv('API_KEY', None)
initial_structure = get_structure('mp-54', api_key)
diff_model = Diffusion(initial_structure)
new_structure = diff_model.sample()
print(new_structure)
