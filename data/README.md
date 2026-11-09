# Dataset

`taiwan_cities.csv` contains 14 city/township representative coordinates on Taiwan's
main island. They are city-centre approximations for teaching the TSP, not routing-grade
coordinates.

| Field | Meaning |
|---|---|
| `name` | City / township label |
| `region` | Rough geographic region |
| `latitude`, `longitude` | WGS 84 decimal degrees |

Distances are computed with the **haversine formula** (great-circle kilometres), not
straight lines in degrees. This keeps the objective meaningful across Taiwan's latitude.

The original notebooks referenced a missing, mojibake-named `.xlsx` file and could not be
reproduced. This CSV replaces it with a transparent, version-controlled dataset.
