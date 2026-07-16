# Lung Nematic

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![NumPy](https://img.shields.io/badge/NumPy-supported-blue)
![SciPy](https://img.shields.io/badge/SciPy-supported-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-app-FF4B4B?logo=streamlit&logoColor=white)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Danpc11/lung-nematic/blob/main/Notebook_Lung_Nematic_Analisys.ipynb)

Pipeline modular para estimar campos nemáticos y detectar defectos topológicos candidatos en imágenes de histología pulmonar.

## Qué genera

Para cada imagen:

- máscara de tejido;
- segmentación nuclear;
- tabla con geometría y orientación de núcleos;
- campo director nemático;
- orden nemático local y global;
- candidatos persistentes `+1/2` y `-1/2`;
- figura de superposición;
- panel diagnóstico;
- métricas en CSV y JSON.

## Estructura

```text
lung_nematic_modular/
├── lung_nematic/
│   ├── config.py
│   ├── io_utils.py
│   ├── preprocessing.py
│   ├── segmentation.py
│   ├── nematic.py
│   ├── defects.py
│   ├── metrics.py
│   ├── visualization.py
│   ├── pipeline.py
│   └── batch.py
├── notebooks/
│   └── Lung_Nematic_Modular_Colab.ipynb
├── config/
│   └── default_config.json
├── metadata_template.csv
├── requirements.txt
└── pyproject.toml
