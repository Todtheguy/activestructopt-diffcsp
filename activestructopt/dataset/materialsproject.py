import requests
from pymatgen.core.structure import Structure

def get_structure(mpid, api_key):
    mp_base_url = "https://api.materialsproject.org/"
    headers = {'accept': 'application/json', 'X-API-KEY': api_key}
    query = {'material_ids': mpid, '_fields': 'structure'}
    response = requests.get(mp_base_url + "materials/summary", 
                             params = query, headers = headers)
    return Structure.from_dict(response.json()['data'][0]['structure'])

