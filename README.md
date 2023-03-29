# Synthaise

Generative sound effects with AI. This project has pivoted to an AI-powered search through a small sound effect library using the UrbanSound8k dataset.

## Dataset

The dataset used in this project can be downloaded from Kaggle at the following link:

[UrbanSound8k Dataset](https://www.kaggle.com/chrisfilo/urbansound8k)

### Instructions to set up the dataset

1. Register or log in to your Kaggle account.
2. Download the `urbansound8k.zip` file from the link above.
3. Extract the contents of the `urbansound8k.zip` file.
4. Place the extracted `UrbanSound8K` folder inside your local repository of the Flask app. The final structure should look like this:
5. Create a folder `audio-files`, this is where the results will be locally stored.

synthaise/
│
├─ templates/
├─ static/
├─ app.py
├─ .gitignore
├─ README.md
├─ audio-files/
└─ UrbanSound8K/
    ├─ audio/
    └─ metadata/

## Running the Flask app locally

1. Make sure you have Python 3.x and `pip` installed on your system.
2. Navigate to the root folder of the Flask app (`synthaise`).
3. Create a virtual environment

```bash
python -m venv venv
```

then:

On Windows:
```bash
venv\Scripts\activate
```

On macOS and Linux:
```bash
source venv/bin/activate

```

5. pip install -r requirements.txt
6. `flask run` will run the development server.

if step 6 this doesn't work, set your environment variables:

On Windows:
```bash
set FLASK_APP=app.py
```

On macOS and Linux:
```bash
export FLASK_APP=app.py
```
