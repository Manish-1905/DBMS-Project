import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Same shape as the MTQ "er" schema.
ER_SCHEMA = {
    "node_props": {
        "User": [{"property": "userId", "datatype": "INTEGER"},
                 {"property": "firstName", "datatype": "STRING"},
                 {"property": "gender", "datatype": "STRING"},
                 {"property": "state", "datatype": "STRING"}],
        "Movie": [{"property": "movieId", "datatype": "INTEGER"},
                  {"property": "title", "datatype": "STRING"},
                  {"property": "year", "datatype": "INTEGER"}],
        "Genre": [{"property": "name", "datatype": "STRING"}],
        "IpAddress": [{"property": "address", "datatype": "STRING"}],
    },
    "rel_props": {"WATCHED": [{"property": "watchCount", "datatype": "INTEGER"}]},
    "relationships": [
        {"start": "User", "type": "WATCHED", "end": "Movie"},
        {"start": "User", "type": "USES", "end": "IpAddress"},
        {"start": "Movie", "type": "HAS", "end": "Genre"},
    ],
}


@pytest.fixture
def er_schema():
    return copy.deepcopy(ER_SCHEMA)
