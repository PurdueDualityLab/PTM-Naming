# EMSE Artifact

> Artifact repository for paper: **"I see models being a whole other thing": Naming Practices of Pre-Trained Models in Hugging Face**

## Overview


- Survey study ($4.1):
  - This part of our study was used to answer *RQ1-3* in the paper.
- Repository mining ($4.2): 
  - This part of our study was used to answer *RQ2, 4* in the paper.
- DARA ($5.4.2):
  - This part of our study was used to answer *RQ4* in the paper.


## Directory Structure
| Top-level folder            | Second-level folder      |
|-----------------------------|--------------------------|
| `Naming_anomaly_detection/` | APTM/                  |
|                             | CL/                    |
|                             | DARA/                  |
|                             | data_files/            |
|                             | tools/                 |
|                             | vector/                |
| `Naming_practices/`         | naming_conventions/    |
|                             | naming_elements/       |
| `survey/`                   | Instruments/           |
|                             | data/                  |


## [Survey Study](/survey) ($IV.A)
This folder includes scripts and data relevant to Table 1, Table 2, Table 3, Table 4, Figure 5, Table 5, Table 6, Table 7, Table 8, Table 9, Table 10 in the paper.


- [Survey Instrument](/survey/Instruments/Survey%20Instrument.pdf): 
  - The survey instrument used in the study. ($IV.A.2)

- [Survey data](/survey/data/survey_data.xlsx): 
  - The raw data of our survey study. ($V)
  
- [Qualitative analysis - data](/survey/data/Survey%20Analysis.xlsx): 
  - The analysis of the qualitative survey data. ($IV.A.4)

- [Qualitative analysis - Themes](/survey/Instruments/PTMVsTraditionalNaming-Themes-v1.docx.pdf): 
  - The extracted themes for the qualitative analysis. ($IV.A.4)

## [Repo mining](/Naming_practices) ($IV.B, $IV.C)
This folder includes scripts and data relevant to Listing 1, Listing 2, Figure 4, Figure 5, Figure 6, Table 5, and Table 6 in the paper.

- [Naming elements analysis](/Naming_practices/naming_elements)
  - The prompts and scripts to analyze the naming elements of the PTMs.
  - The manually labeled groud truth data for evaluation.

- [Naming convention analysis](/Naming_practices/naming_conventions)
  - The prompts and scripts to analyze the naming conventions of the PTMs.
  - The manually labeled groud truth data for evaluation.

## [Naming_anomaly_detection](/Naming_anomaly_detection) ($V.B)
This folder includes scripts and data relevant to Table 11 in the paper.


- [APTMs](/Naming_anomaly_detection/APTM)
  - The APTM model and graph conversion pipeline.
- [CL](/Naming_anomaly_detection/CL)
  - The contrastive learning solution for detecting naming anomalies.
- [DARA](/Naming_anomaly_detection/DARA)
  - The DNN Architecture Assessment pipeline.
- [Data files](/Naming_anomaly_detection/data_files)
  - The data collections scripts files used in the study.
  - The collected data files used in the study.
- [Tools](/Naming_anomaly_detection/tools)
  - The utils used in the study.
- [Vector](/Naming_anomaly_detection/vector)
  - The feature extractors used in the study.
