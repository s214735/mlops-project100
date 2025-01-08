# p100
Your virtual PokéDex that recognizes Pokémon you meet on your way as well as their type, and recommends you a Pokémon type that will have the advantage in combat.

Overall goal of the project:
The goal of this project is to develop a machine learning pipeline for recognizing Pokémon species from images and identifying their types. Once identified, the model will recommend an optimal counter type based on a type-effectiveness matrix. This system will function as a virtual PokéDex, which not only provides information on Pokémon but also offers battle strategies by recommending types with advantages in combat scenarios.

Frameworks:
For the project structure, we are using the CookieCutter framework, specifically a custom MLOps template from the course (02476). The template provides a robust foundation for managing and organizing the MLOps pipeline, ensuring efficient code management, version control, and scalability. This framework will help structure the project and give a better overview of its different segments.

Data:
The dataset used in this project is sourced from Kaggle and contains approximately 40 images per 1,000 Pokémon species. These images are structured in subdirectories according to the species class, with each image resized to 128x128 pixels and stored in PNG format. The dataset was originally created for a Flutter-based PokéDex project. It has been split into 80% training, 10% testing, and 10% evaluation data, with an option to modify the splits using the generate_splits.py script. In addition to the images, metadata is provided for each Pokémon, including base stats such as HP, Attack, Speed, Defense, Special Attack, and Special Defense. The dataset is further enriched by the addition of type information for each Pokémon species. Data augmentation is highly recommended to improve model performance.

Model:
We are using a pre-trained model based on ResNet (from PyTorch) for the task of image recognition. This model is a CNN, based on a normal neural network, but with residual blocks. It will be fine-tuned to classify Pokémon images and predict their corresponding types. Once the model classifies a Pokémon and identifies its type, the type-effectiveness matrix will be applied to recommend a counter type based on combat advantages. PyTorchLightning is also used to add more structure and optimize the code.

Final considerations:
We are also utilizing Weights and Biases for tracking experiments, monitoring model performance, and visualizing results. The integration of CookieCutter, the ResNet pre-trained model, and Weights and Biases will provide a comprehensive and efficient workflow for the project.


## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
