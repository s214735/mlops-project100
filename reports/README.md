# Exam template for 02476 Machine Learning Operations

This is the report template for the exam. Please only remove the text formatted as with three dashes in front and behind
like:

```--- question 1 fill here ---```

Where you instead should add your answers. Any other changes may have unwanted consequences when your report is
auto-generated at the end of the course. For questions where you are asked to include images, start by adding the image
to the `figures` subfolder (please only use `.png`, `.jpg` or `.jpeg`) and then add the following code in your answer:

```markdown
![my_image](figures/<image>.<extension>)
```

In addition to this markdown file, we also provide the `report.py` script that provides two utility functions:

Running:

```bash
python report.py html
```

Will generate a `.html` page of your report. After the deadline for answering this template, we will auto-scrape
everything in this `reports` folder and then use this utility to generate a `.html` page that will be your serve
as your final hand-in.

Running

```bash
python report.py check
```

Will check your answers in this template against the constraints listed for each question e.g. is your answer too
short, too long, or have you included an image when asked. For both functions to work you mustn't rename anything.
The script has two dependencies that can be installed with

```bash
pip install typer markdown
```

## Overall project checklist

The checklist is *exhaustive* which means that it includes everything that you could do on the project included in the
curriculum in this course. Therefore, we do not expect at all that you have checked all boxes at the end of the project.
The parenthesis at the end indicates what module the bullet point is related to. Please be honest in your answers, we
will check the repositories and the code to verify your answers.

### Week 1

* [x] Create a git repository (M5)
* [x] Make sure that all team members have write access to the GitHub repository (M5)
* [x] Create a dedicated environment for you project to keep track of your packages (M2)
* [x] Create the initial file structure using cookiecutter with an appropriate template (M6)
* [x] Fill out the `data.py` file such that it downloads whatever data you need and preprocesses it (if necessary) (M6)
* [x] Add a model to `model.py` and a training procedure to `train.py` and get that running (M6)
* [x] Remember to fill out the `requirements.txt` and `requirements_dev.txt` file with whatever dependencies that you
    are using (M2+M6)
* [x] Remember to comply with good coding practices (`pep8`) while doing the project (M7)
* [(x)] Do a bit of code typing and remember to document essential parts of your code (M7)
* [x] Setup version control for your data or part of your data (M8)
* [x] Add command line interfaces and project commands to your code where it makes sense (M9)
* [x] Construct one or multiple docker files for your code (M10)
* [x] Build the docker files locally and make sure they work as intended (M10)
* [x] Write one or multiple configurations files for your experiments (M11)
* [x] Used Hydra to load the configurations and manage your hyperparameters (M11)
* [x] Use profiling to optimize your code (M12)
* [x] Use logging to log important events in your code (M14)
* [x] Use Weights & Biases to log training progress and other important metrics/artifacts in your code (M14)
* [x] Consider running a hyperparameter optimization sweep (M14)
* [(x)] Use PyTorch-lightning (if applicable) to reduce the amount of boilerplate in your code (M15)

### Week 2

* [x] Write unit tests related to the data part of your code (M16)
* [x] Write unit tests related to model construction and or model training (M16)
* [ ] Calculate the code coverage (M16)
* [x] Get some continuous integration running on the GitHub repository (M17)
* [x] Add caching and multi-os/python/pytorch testing to your continuous integration (M17)
* [(x)] Add a linting step to your continuous integration (M17)
* [x] Add pre-commit hooks to your version control setup (M18)
* [ ] Add a continues workflow that triggers when data changes (M19)
* [x] Add a continues workflow that triggers when changes to the model registry is made (M19)
* [x] Create a data storage in GCP Bucket for your data and link this with your data version control setup (M21)
* [ ] Create a trigger workflow for automatically building your docker images (M21)
* [x] Get your model training in GCP using either the Engine or Vertex AI (M21)
* [x] Create a FastAPI application that can do inference using your model (M22)
* [x] Deploy your model in GCP using either Functions or Run as the backend (M23)
* [(x)] Write API tests for your application and setup continues integration for these (M24)
* [(x) ] Load test your application (M24)
* [x] Create a more specialized ML-deployment API using either ONNX or BentoML, or both (M25)
* [x] Create a frontend for your API (M26)

### Week 3

* [ ] Check how robust your model is towards data drifting (M27)
* [ ] Deploy to the cloud a drift detection API (M27)
* [ ] Instrument your API with a couple of system metrics (M28)
* [ ] Setup cloud monitoring of your instrumented application (M28)
* [ ] Create one or more alert systems in GCP to alert you if your app is not behaving correctly (M28)
* [x] If applicable, optimize the performance of your data loading using distributed data loading (M29)
* [x] If applicable, optimize the performance of your training pipeline by using distributed training (M30)
* [x] Play around with quantization, compilation and pruning for you trained models to increase inference speed (M31)

### Extra

* [ ] Write some documentation for your application (M32)
* [ ] Publish the documentation to GitHub Pages (M32)
* [ ] Revisit your initial project description. Did the project turn out as you wanted?
* [ ] Create an architectural diagram over your MLOps pipeline
* [ ] Make sure all group members have an understanding about all parts of the project
* [ ] Uploaded all your code to GitHub

## Group information

### Question 1
> **Enter the group number you signed up on <learn.inside.dtu.dk>**
>
> Answer:

100

### Question 2
> **Enter the study number for each member in the group**
>
> Example:
>
> *sXXXXXX, sXXXXXX, sXXXXXX*
>
> Answer:

s214735, s214742, s214739, s214733, s214731

### Question 3
> **A requirement to the project is that you include a third-party package not covered in the course. What framework**
> **did you choose to work with and did it help you complete the project?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We used the third-party framework ... in our project. We used functionality ... and functionality ... from the*
> *package to do ... and ... in our project*.
>
> Answer:

In our project, we used two third-party frameworks, pandas and UMAP, which were essential for our tasks. We employed pandas to load and manage Pokémon metadata from a Kaggle dataset. This framework provided data operations like sorting and filtering, which were necessary for preparing our data.

Additionally, we integrated umap for visualizing the vector space of our model. umap helped in reducing the dimensionality of our data, allowing us to visually inspect how different Pokémon characteristics clustered together. This visualization aided in assessing the effectiveness of our data modeling approach.

Using pandas and umap allowed us to efficiently process and analyze our data, providing clear insights into the model. 

## Coding environment

> In the following section we are interested in learning more about you local development environment. This includes
> how you managed dependencies, the structure of your code and how you managed code quality.

### Question 4

> **Explain how you managed dependencies in your project? Explain the process a new team member would have to go**
> **through to get an exact copy of your environment.**
>
> Recommended answer length: 100-200 words
>
> Example:
> *We used ... for managing our dependencies. The list of dependencies was auto-generated using ... . To get a*
> *complete copy of our development environment, one would have to run the following commands*
>
> Answer:

In our project, we managed our dependencies using two key files: `requirements.txt` and `requirements_dev.txt`. The `requirements.txt` file contains all the essential libraries needed to run the project, ensuring that any core functionality works seamlessly across different setups. On the other hand, `requirements_dev.txt` includes additional packages that are necessary for development purposes, such as testing and debugging tools.

For a new team member to set up an exact copy of our environment, they would need to follow these steps:

1. Clone the project repository from our version control system to their local machine.
2. Ensure that Python is installed on their system.
3. Create a virtual environment (venv) to isolate and manage project-specific dependencies. This can be done by running `python -m venv env` in the project's root directory.
4. Activate the virtual environment using `source env/bin/activate` on Unix or macOS, or `env\Scripts\activate` on Windows.
5. Run `pip install -r requirements.txt` to install all the required libraries listed in the `requirements.txt` file.
6. Optionally, if they are involved in development, they should also run `pip install -r requirements_dev.txt` to install the additional development dependencies.

By following these steps, a new team member can quickly set up the project environment, ensuring they have all the necessary dependencies to start working on the project immediately.


### Question 5

> **We expect that you initialized your project using the cookiecutter template. Explain the overall structure of your**
> **code. What did you fill out? Did you deviate from the template in some way?**
>
> Recommended answer length: 100-200 words
>
> Example:
> *From the cookiecutter template we have filled out the ... , ... and ... folder. We have removed the ... folder*
> *because we did not use any ... in our project. We have added an ... folder that contains ... for running our*
> *experiments.*
>
> Answer:

We initialized our project using the cookiecutter template and adapted it to fit our specific needs. The primary directories we filled out and utilized are:

- **`config` folder**: This directory holds all configuration files that manage settings and parameters used across different parts of our project, ensuring consistency and ease of adjustments.
- **`data` folder**: Used for storing raw and processed data sets. This organization helps in managing the data lifecycle within our project.
- **`dockerfiles` folder**: Contains Dockerfiles which are crucial for creating Docker containers that ensure our environment is consistent across all development and production systems.
- **`model` folder**: This is where we store model definitions and training scripts, centralizing all machine learning algorithms and models.
- **`src` folder**: The source code for our application is kept here, including all the primary Python scripts and modules.
- **`tests` folder**: Dedicated to housing test cases and scripts, which are vital for ensuring the robustness and reliability of our code through continuous integration practices.

Additionally, we maintained a structured approach to version control using Git, which included setting up `.gitignore` files to exclude unnecessary files from being tracked and `workflows` for automating tasks such as testing, building, and deploying using GitHub Actions.

By carefully organizing these folders and incorporating essential development practices, we created a project structure that is both scalable and easy to navigate for any new team members or stakeholders.

### Question 6

> **Did you implement any rules for code quality and format? What about typing and documentation? Additionally,**
> **explain with your own words why these concepts matters in larger projects.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We used ... for linting and ... for formatting. We also used ... for typing and ... for documentation. These*
> *concepts are important in larger projects because ... . For example, typing ...*
>
> Answer:

In our project, we used `ruff` to lint our code before commits, ensuring it was error-free and maintained a consistent style. We documented each function with docstrings that explained its purpose, inputs, and outputs, and we included comments to clarify complex code sections.

We also employed type annotations throughout our code to specify expected input and output types, helping to prevent type-related errors and making the code clearer for other developers.

These practices are crucial in larger projects as they improve code quality, enhance readability, and simplify maintenance. They also facilitate collaboration among developers by making the codebase easier to understand and navigate, which is vital as projects grow and evolve.

## Version control

> In the following section we are interested in how version control was used in your project during development to
> corporate and increase the quality of your code.

### Question 7

> **How many tests did you implement and what are they testing in your code?**
>
> Recommended answer length: 50-100 words.
>
> Example:
> *In total we have implemented X tests. Primarily we are testing ... and ... as these the most critical parts of our*
> *application but also ... .*
>
> Answer:

In our project, we implemented a suite of tests, organized into four main files: `test_api.py`, `test_data.py`, `test_model.py`, and `test_train.py`. These tests are automated through GitHub workflows and executed within GitHub Actions to ensure continuous integration and consistent code quality.

- **`test_api.py`**: Tests the API to make sure all endpoints respond correctly.
- **`test_data.py`**: Checks that data loads and processes correctly, preserving format and accuracy.
- **`test_model.py`**: Ensures our models are predicting the correct format 
- **`test_train.py`**: Confirms that our dataloader is correctly loading data and that our model is training as expected.

These automated tests help us catch issues early, keeping our application reliable as we continuously update it.

### Question 8

> **What is the total code coverage (in percentage) of your code? If your code had a code coverage of 100% (or close**
> **to), would you still trust it to be error free? Explain you reasoning.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *The total code coverage of code is X%, which includes all our source code. We are far from 100% coverage of our **
> *code and even if we were then...*
>
> Answer:

In our project, we currently have a code coverage of approximately 60-70%. Achieving 100% code coverage is often viewed as ideal, but it's important to recognize that even this level does not guarantee an error-free application. Code coverage only measures the extent of code executed during tests, not the effectiveness or comprehensiveness of the testing itself. While a high code coverage can give a false sense of security, especially if not all use cases and edge cases are tested, we did not think achieving 100% coverage was worth the time. Instead, we focused on covering 60-70% of the code, prioritizing the most crucial aspects to detect bugs and errors early when making changes to the code and model. This approach allowed us to efficiently balance resource allocation with the need for effective error detection in key areas of our application.

### Question 9

> **Did you workflow include using branches and pull requests? If yes, explain how. If not, explain how branches and**
> **pull request can help improve version control.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We made use of both branches and PRs in our project. In our group, each member had an branch that they worked on in*
> *addition to the main branch. To merge code we ...*
>
> Answer:

In our project, we used Git for version control, which was a new experience for many of us. We organized our work by creating specific branches for different features, allowing team members to work independently or in small groups on separate parts of the project. This approach helped keep the main code stable as we developed new features.

For merging these features into the main code, we used pull requests. This process allowed everyone on the team to review and discuss the changes before they were added to the main branch, ensuring the code was both high-quality and compatible with existing features.

Sometimes, we made urgent fixes directly on the main branch when immediate changes were needed that affected the whole project. Overall, using branches and pull requests made it easier to manage our project and keep our code reliable.

### Question 10

> **Did you use DVC for managing data in your project? If yes, then how did it improve your project to have version**
> **control of your data. If no, explain a case where it would be beneficial to have version control of your data.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did make use of DVC in the following way: ... . In the end it helped us in ... for controlling ... part of our*
> *pipeline*
>
> Answer:

In our project, we didn’t use DVC for data management due to issues with Google’s policies for integrating it with Google Drive. Instead, we used Google Cloud for storing our data and accessing their GPUs for training. For version control, we relied on Git, but only for managing our code, not our data.

Version control for data would have been beneficial in cases where datasets are updated or modified frequently. For example, if new data is added or existing data is cleaned, a version control system would allow us to track changes, revert to previous versions, and ensure reproducibility. While Git helped us manage our code effectively, having a similar system for data would have made managing updates and collaborations more efficient and organized.

### Question 11

> **Discuss you continuous integration setup. What kind of continuous integration are you running (unittesting,**
> **linting, etc.)? Do you test multiple operating systems, Python  version etc. Do you make use of caching? Feel free**
> **to insert a link to one of your GitHub actions workflow.**
>
> Recommended answer length: 200-300 words.
>
> Example:
> *We have organized our continuous integration into 3 separate files: one for doing ..., one for running ... testing*
> *and one for running ... . In particular for our ..., we used ... .An example of a triggered workflow can be seen*
> *here: <weblink>*
>
> Answer:

In our project, we set up workflows to ensure our code and models worked correctly. These workflows checked that our data was in the correct format, verified that our model could initialize properly, and ensured that prediction speeds did not slow down over time.

We also created a testing matrix in our workflows to test across multiple operating systems and Python versions. Specifically, we tested on the following:

- Operating systems: `ubuntu-latest`, `windows-latest`, `macos-latest`
- Python version: `3.11`

To optimize workflow runtimes, we implemented caching for dependencies. This allowed GitHub to reuse previously installed requirements and only install new ones when necessary, significantly speeding up our runs and avoid installing the same dependencies multiple times.

Additionally, we tried to create a model testing stage that would be triggered by Weights & Biases (wandb) when adding a model to the registry. This stage was meant to validate new models before integrating them. However, we encountered many issues with this setup and spent a lot of time trying to fix it. In the end, we decided to remove it because we couldn’t get it to work as intended.

Overall, these workflows helped us identify issues early, maintain stable performance across platforms, and improve efficiency with caching, ensuring our project development was smooth and reliable.

## Running code and tracking experiments

> In the following section we are interested in learning more about the experimental setup for running your code and
> especially the reproducibility of your experiments.

### Question 12

> **How did you configure experiments? Did you make use of config files? Explain with coding examples of how you would**
> **run a experiment.**
>
> Recommended answer length: 50-100 words.
>
> Example:
> *We used a simple argparser, that worked in the following way: Python  my_script.py --lr 1e-3 --batch_size 25*
>
> Answer:

We used config files and Hydra to manage and run our experiments. Hydra made it simple to change settings without altering the main code. For example, to run an experiment with a specific batch size, we used the following command:

`python src/p100/train.py train.batch_size=64`

This allowed us to quickly test different configurations by changing parameters directly in the command line, making our experiments more flexible and organized.

### Question 13

> **Reproducibility of experiments are important. Related to the last question, how did you secure that no information**
> **is lost when running experiments and that your experiments are reproducible?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We made use of config files. Whenever an experiment is run the following happens: ... . To reproduce an experiment*
> *one would have to do ...*
>
> Answer:

We ensured the reproducibility of our experiments by using Weights & Biases to log as many training parameters as possible. This included hyperparameters, model architecture details, training progress, and results. By logging these parameters, we could easily review the settings and outcomes of any experiment, making it straightforward to reproduce the training process when needed.

In addition to wandb, we used configuration files to define default settings for our experiments. These config files contained all the key parameters, such as learning rates, batch sizes, and data paths, ensuring that experiments could be run with consistent baseline settings. Any modifications to these defaults were logged in wandb, so changes were fully traceable.

To reproduce an experiment, we simply looked up the parameters in Weights & Biases and used the same parameters to run the experiment again. By logging the perfomance we could also compare the results of the new experiment with the original one, ensuring that the results were consistent.

### Question 14

> **Upload 1 to 3 screenshots that show the experiments that you have done in W&B (or another experiment tracking**
> **service of your choice). This may include loss graphs, logged images, hyperparameter sweeps etc. You can take**
> **inspiration from [this figure](figures/wandb.png). Explain what metrics you are tracking and why they are**
> **important.**
>
> Recommended answer length: 200-300 words + 1 to 3 screenshots.
>
> Example:
> *As seen in the first image when have tracked ... and ... which both inform us about ... in our experiments.*
> *As seen in the second image we are also tracking ... and ...*
>
> Answer:

--- question 14 fill here ---

### Question 15

> **Docker is an important tool for creating containerized applications. Explain how you used docker in your**
> **experiments/project? Include how you would run your docker images and include a link to one of your docker files.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For our project we developed several images: one for training, inference and deployment. For example to run the*
> *training docker image: `docker run trainer:latest lr=1e-3 batch_size=64`. Link to docker file: <weblink>*
>
> Answer:

--- question 15 fill here ---

### Question 16

> **When running into bugs while trying to run your experiments, how did you perform debugging? Additionally, did you**
> **try to profile your code or do you think it is already perfect?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *Debugging method was dependent on group member. Some just used ... and others used ... . We did a single profiling*
> *run of our main code at some point that showed ...*
>
> Answer:

We primarily used the Visual Studio Code (VSC) debugger to troubleshoot issues in our code. While the debugger was useful for identifying errors in the local codebase, we often found it less helpful when dealing with bugs related to server connections. Many of the challenges we faced involved integrating different services and ensuring communication between servers, which required a different approach to debugging.

For server-related issues, we relied heavily on logging and inspecting error messages to understand what was going wrong. This approach helped us identify the root cause of many problems and find solutions more effectively.

We did not perform extensive profiling of our code, as our focus was on functionality rather than optimization at this stage. However, we acknowledge that profiling could help us improve the performance of our code in future iterations.

## Working in the cloud

> In the following section we would like to know more about your experience when developing in the cloud.

### Question 17

> **List all the GCP services that you made use of in your project and shortly explain what each service does?**
>
> Recommended answer length: 50-200 words.
>
> Example:
> *We used the following two services: Engine and Bucket. Engine is used for... and Bucket is used for...*
>
> Answer:

Generally we have tried to use all of the GCP services provided to us in the course material. In the end we used the virtual machines, which allows us the create instances that would induce the ability to train on remote GPUs. We also used the cloud build services, particularly setting a trigger so that it runs a cloudbuild workflow each time code is pushed to main. Another great service we used was the cloud storage, specifically the buckets which allows us to store data remotely.

### Question 18

> **The backbone of GCP is the Compute engine. Explained how you made use of this service and what type of VMs**
> **you used?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We used the compute engine to run our ... . We used instances with the following hardware: ... and we started the*
> *using a custom container: ...*
>
> Answer:



### Question 19

> **Insert 1-2 images of your GCP bucket, such that we can see what data you have stored in it.**
> **You can take inspiration from [this figure](figures/bucket.png).**
>
> Answer:

--- question 19 fill here ---

### Question 20

> **Upload 1-2 images of your GCP artifact registry, such that we can see the different docker images that you have**
> **stored. You can take inspiration from [this figure](figures/registry.png).**
>
> Answer:

--- question 20 fill here ---

### Question 21

> **Upload 1-2 images of your GCP cloud build history, so we can see the history of the images that have been build in**
> **your project. You can take inspiration from [this figure](figures/build.png).**
>
> Answer:

--- question 21 fill here ---

### Question 22

> **Did you manage to train your model in the cloud using either the Engine or Vertex AI? If yes, explain how you did**
> **it. If not, describe why.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We managed to train our model in the cloud using the Engine. We did this by ... . The reason we choose the Engine*
> *was because ...*
>
> Answer:

--- question 22 fill here ---

## Deployment

### Question 23

> **Did you manage to write an API for your model? If yes, explain how you did it and if you did anything special. If**
> **not, explain how you would do it.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did manage to write an API for our model. We used FastAPI to do this. We did this by ... . We also added ...*
> *to the API to make it more ...*
>
> Answer:

--- question 23 fill here ---

### Question 24

> **Did you manage to deploy your API, either in locally or cloud? If not, describe why. If yes, describe how and**
> **preferably how you invoke your deployed service?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For deployment we wrapped our model into application using ... . We first tried locally serving the model, which*
> *worked. Afterwards we deployed it in the cloud, using ... . To invoke the service an user would call*
> *`curl -X POST -F "file=@file.json"<weburl>`*
>
> Answer:

--- question 24 fill here ---

### Question 25

> **Did you perform any unit testing and load testing of your API? If yes, explain how you did it and what results for**
> **the load testing did you get. If not, explain how you would do it.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For unit testing we used ... and for load testing we used ... . The results of the load testing showed that ...*
> *before the service crashed.*
>
> Answer:

--- question 25 fill here ---

### Question 26

> **Did you manage to implement monitoring of your deployed model? If yes, explain how it works. If not, explain how**
> **monitoring would help the longevity of your application.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did not manage to implement monitoring. We would like to have monitoring implemented such that over time we could*
> *measure ... and ... that would inform us about this ... behaviour of our application.*
>
> Answer:

--- question 26 fill here ---

## Overall discussion of project

> In the following section we would like you to think about the general structure of your project.

### Question 27

> **How many credits did you end up using during the project and what service was most expensive? In general what do**
> **you think about working in the cloud?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *Group member 1 used ..., Group member 2 used ..., in total ... credits was spend during development. The service*
> *costing the most was ... due to ... . Working in the cloud was ...*
>
> Answer:

--- question 27 fill here ---

### Question 28

> **Did you implement anything extra in your project that is not covered by other questions? Maybe you implemented**
> **a frontend for your API, use extra version control features, a drift detection service, a kubernetes cluster etc.**
> **If yes, explain what you did and why.**
>
> Recommended answer length: 0-200 words.
>
> Example:
> *We implemented a frontend for our API. We did this because we wanted to show the user ... . The frontend was*
> *implemented using ...*
>
> Answer:

--- question 28 fill here ---

### Question 29

> **Include a figure that describes the overall architecture of your system and what services that you make use of.**
> **You can take inspiration from [this figure](figures/overview.png). Additionally, in your own words, explain the**
> **overall steps in figure.**
>
> Recommended answer length: 200-400 words
>
> Example:
>
> *The starting point of the diagram is our local setup, where we integrated ... and ... and ... into our code.*
> *Whenever we commit code and push to GitHub, it auto triggers ... and ... . From there the diagram shows ...*
>
> Answer:

--- question 29 fill here ---

### Question 30

> **Discuss the overall struggles of the project. Where did you spend most time and what did you do to overcome these**
> **challenges?**
>
> Recommended answer length: 200-400 words.
>
> Example:
> *The biggest challenges in the project was using ... tool to do ... . The reason for this was ...*
>
> Answer:

--- question 30 fill here ---

### Question 31

> **State the individual contributions of each team member. This is required information from DTU, because we need to**
> **make sure all members contributed actively to the project. Additionally, state if/how you have used generative AI**
> **tools in your project.**
>
> Recommended answer length: 50-300 words.
>
> Example:
> *Student sXXXXXX was in charge of developing of setting up the initial cookie cutter project and developing of the*
> *docker containers for training our applications.*
> *Student sXXXXXX was in charge of training our models in the cloud and deploying them afterwards.*
> *All members contributed to code by...*
> *We have used ChatGPT to help debug our code. Additionally, we used GitHub Copilot to help write some of our code.*
> Answer:
- Student s214731 has been in charge of setting up the GCP, experimenting with virtual machines, and how to implement automatic building of Docker images.
- Student s214733 has been in charge of writing the unit tests, and creating the continuous integration that uses GitHub actions. Furthermore, the student has been helping with setting up the API.
- Student s214735 has been in charge of creating our model as well as setting up the API backend/frontend. Furthermore, the student has worked on deployment of the model in GCP, and created pre-commits in the version control setup. 
- Student s214739 has been in charge of data loading and processing, and has also been helping a lot with setting up the cloud (especially the GCP Bucket). Furthermore, the student has spent time on the model training.
- Student s214742 has been in charge of setting up the profiling and logging, creating the WANDB project and has also worked on the continuous integration. The student has spent time on config files and data loading as well. 
- All members contributed to code by working on all parts of the project. Every team member has been involved to some degree in all parts of the project, and knows how the different operations work.
- We have used ChatGPT to help debug and/or write some parts of our code.

