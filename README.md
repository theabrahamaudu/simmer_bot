Simmer Bot
==============================

[[Actual Documentation Incoming]]

Project Organization
------------
Tree Structures:
- [Overall](./dir_tree.md)
- [Frontend](./frontend/dir_tree_frontend.md)
- [Backend](./backend/dir_tree_backend.md)

## System Architecture
Below is the project architecture. Each shaded area represents a separate docker container.

![image](https://github.com/theabrahamaudu/Multi-Platfrom_Job_Recommender/blob/main/backend/docs/architecture.png)

## Quick Start

Spin up an instannce of the Job Recommender app on your local machine in a few simple steps:
##### N.B: Requires Docker on Ubuntu (WSL for Windows) [7.5GB vRAM Minimum]

- Clone this repository
    ~~~
    git clone https://github.com/theabrahamaudu/Multi-Platfrom_Job_Recommender.git
    ~~~
- Create a virtual environment
- Install `transformers` package
    ~~~
    pip install transformers==4.35.2
    ~~~
- Download recommender model
    ~~~
    cd backend && python src/utils/get_model.py
    ~~~
- Build and spin up Docker containers
    ~~~
    cd .. && COMPOSE_DOCKER_CLI_BUILD=1 DOCKER_BUILDKIT=1 docker compose up -d
    ~~~
- Open the URL of the frontend container via the Docker Desktop  

## Description
The central idea behind this app is to reduce the need to search across multiple websites searching for jobs that are a good fit. At the core of it, it does not disenfranchise the websites from which these jobs are found, as all search hits will redirect the user to the original source when the user decides to apply.

The app works by scraping jobs from websites such as LinkedIn, Indeed and Jobberman. These jobs are then saved to a Cassandra database, afterwhich the jobs are vectorized with a language model and stored as vectors on a Chroma database. Duplicates are prevented by performing an assertion check on the hashed job links. In deployment, the scrapers run as part of a `cron job` everyday at 2am to grow the jobs database. As part of maintaining recency, the scraping pipeline cleans up jobs older than 30 days at the end of the process.

Users are allowed to create unique profiles which capture their skills, work history and user defined preferences. In addition, the website tracks implicit preferences by storing search history and search result interactions. All these data points are used to generate characteristic user vectors to aid in searching for the best fit jobs.

The recommender system is `content-based`. During the search process, the search query along with the user data and activity metadata are combined to form a composite query which is then vectorised and searched against the job vectors index to yield the top 10 job identifiers. These identifiers are then used to pull the jobs data from the Cassandra database for use by the user interface.

The user interface is lightweight, having a 
- Home page  
- Search page, and
- Profile page. 

The home page allows the user to login, and afterwards, the user can logout from any page. The search page automatically presents recommended jobs even before searching, based on the user's profile. The profile page allows the user to update their information, change their password and delete their account.


## Basic Workflow
- Setup job scrapers
- Design Cassandra database schema and tables
- Setup Chroma collection
- Setup IO logic for databases
- Setup CRUD endpoints
- Integrate scrapers with database
- Setup index search logic
- Frontend design

## API Structure
- `/admin` : Admin user endpoints
- `/users` : User data endpoints
- `/jobs` : Jobs text data endpoint
- `/clicks` : Clicks (interaction) metadata endpoint
- `/search` : Search query metadata endpoints
- `/index` : Job vectors endpoints

## Dependencies
#### OS Level
- Ubuntu (Developed and tested on 22.04)
- Docker (Use WSL mode if on Windows) [7.5GB vRAM Minimum]
- Firefox (Headless mode)
- Gecko Driver

#### Python (3.10.12)
Backend:

    click
    Sphinx
    coverage
    flake8
    python-dotenv>=0.5.1
    pandas==2.1.2
    selenium==4.14.0
    beautifulsoup4==4.12.2
    ipykernel==6.26.0
    cassandra-driver==3.28.0
    pyyaml==6.0.1
    chromadb==0.4.18
    pydantic==1.10.13
    fastapi==0.104.1
    uvicorn==0.24.0.post1
    sentence-transformers==2.2.2
    pytest==7.4.3
    pytest-cov==4.1.0
    directory_tree

Frontend:

    click
    Sphinx
    coverage
    flake8
    python-dotenv>=0.5.1
    streamlit==1.28.2
    streamlit-extras
    pyyaml==6.0.1
    pytest==7.4.3
    pytest-cov==4.1.0

## Installing
Refer to the [Quick Start](#quick-start) section.

## Miscellaneous
**Testing**  
To run tests on each container, use Docker exec
- Backend
    ~~~
    docker exec backend pytest --cov=. tests/
    ~~~
- Frontend
    ~~~
    docker exec frontend pytest --cov=. tests/
    ~~~

-----------------

**Generating Directory Trees**    

Available on local machine setup  

- Overall tree
    ~~~
    python backend/src/utils/dir_tree_gen.py
    ~~~
- Frontend tree
    ~~~
    python backend/src/utils/dir_tree_gen.py frontend
    ~~~
- Backend tree  
    ~~~
    python backend/src/utils/dir_tree_gen.py backend
    ~~~
## Help
Feel free to reach out to me or create a new issue if you encounter any problems setting up or using the Job Recommender app.

## Possible Improvements/Ideas

- [ ] More tests
- [ ] Custom collaborative model when substantial user data is available
- [ ] Sleeker UI/UX

## Authors

Contributors names and contact info

*Abraham Audu*

* GitHub - [@the_abrahamaudu](https://github.com/theabrahamaudu)
* X (formerly Twitter) - [@the_abrahamaudu](https://x.com/the_abrahamaudu)
* LinkedIn - [@theabrahamaudu](https://www.linkedin.com/in/theabrahamaudu/)
* Instagram - [@the_abrahamaudu](https://www.instagram.com/the_abrahamaudu/)
* YouTube - [@DataCodePy](https://www.youtube.com/@DataCodePy)

## Version History

* See [commit change](https://github.com/theabrahamaudu/Multi-Platfrom_Job_Recommender/commits/main/)
* See [release history](https://github.com/theabrahamaudu/Multi-Platfrom_Job_Recommender/releases)

## Acknowledgments

* This project was inspired by iNeuron Internship Platform
* Cassandra documentation by DataStax
* Chroma documentation
* This [video](https://youtu.be/4Zy90rd0bkU?si=IAodvhHb7KmigtoC) helped me with CRUD logic
* I got a lot of help from different websites in debugging at different stages (StackOverflow, etc)