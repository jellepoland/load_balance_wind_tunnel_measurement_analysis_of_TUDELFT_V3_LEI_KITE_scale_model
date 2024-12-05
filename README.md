# Load Balance Analysis
This repository contains code that transforms raw data into tables and plots used in a paper discussing the results. The raw data is from a subscale rigid leading-edge inflatable kite model measured during a Wind Tunnel campaign in the Open-Jet Facility of the TU Delft in April 2024. The data is published Open-Source and available on ... [link] The paper is titled: "Wind Tunnel Load Measurements Of A Rigid Subscale Leading Edge Inflatable Kite" published Open-Source in Wind Energy science, [link].

## Installation Instructions
1. Clone the repository:
    ```bash
    git clone https://github.com/jellepoland/load_balance_wind_tunnel_measurement_analysis_of_TUDELFT_V3_LEI_KITE_scale_model
    ```

2. Navigate to the repository folder:
    ```bash
    cd load_balance_wind_tunnel_measurement_analysis_of_TUDELFT_V3_LEI_KITE_scale_model
    ```
    
3. Create a virtual environment:
   
   Linux or Mac:
    ```bash
    python3 -m venv venv
    ```
    
    Windows:
    ```bash
    python -m venv venv
    ```
    
5. Activate the virtual environment:

   Linux or Mac:
    ```bash
    source venv/bin/activate
    ```

    Windows
    ```bash
    .\venv\Scripts\activate
    ```

6. Install the required dependencies:

   For users:
    ```bash
    pip install .
    ```
        
   For developers:
    ```bash
    pip install -e .[dev]
    ```

7. To deactivate the virtual environment:
    ```bash
    deactivate
    ```
### Dependencies
- numpy
- pandas>=1.5.3
- matplotlib>=3.7.1,
- ipykernel,
- statsmodels,
- VSM @ git+https://github.com/ocayon/Vortex-Step-Method.git@develop


## Usages
...

## Citation
If you use this project in your research, please consider citing it. 
Citation details can be found in the [CITATION.cff](CITATION.cff) file included in this repository.
- UPDATE THE CITATION


## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## :warning: License and Waiver

Specify the license under which your software is distributed and include the copyright notice:

> Technische Universiteit Delft hereby disclaims all copyright interest in the program “NAME PROGRAM” (one line description of the content or function) written by the Author(s).
> 
> Prof.dr. H.G.C. (Henri) Werij, Dean of Aerospace Engineering
> 
> Copyright (c) [YEAR] [NAME SURNAME].

## :gem: Help and Documentation
[AWE Group | Developer Guide](https://awegroup.github.io/developer-guide/)


