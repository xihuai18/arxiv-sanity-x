
# arxiv-sanity-X

Arxiv-sanity-X: A robust enhancement of the [arXiv-sanity-lite](https://github.com/karpathy/arxiv-sanity-lite) website that significantly expedites the process of academic research. It diligently polls the arXiv API for new papers, enabling users to conveniently tag papers of interest. Based on SVMs over tf-idf features of paper abstracts, it recommends novel papers for each tag. Users can set up keywords that the system will automatically track, enabling a personalized academic journey. The elegant web user interface allows for seamless searching, ranking, sorting, and analysis of results. Uniquely, Arxiv-sanity-X can dispatch daily emails containing recommendations for new papers based on your tags and keywords. Manage your tags and keywords, stay abreast with recent papers in your area, and never miss an important update!

![Screenshot](arxiv-sanity-x.png)

## Table of Contents
1. [To Run](#To-Run)
2. [Documents](#Documents)


## To Run

### Requirements

 Install via requirements:

 ```bash
#  requires python <= 3.9
 pip install -r requirements.txt
 ```

### Configurate
In the first place, create and setup your config in `vars.py`, a template is in `vars_template.py`:
```python
# database
DATA_DIR = "data" # !put it on an ssd for speed
# email
from_email = "your_email@mail.com"
smtp_server = "smtp.mail.com"
smtp_port = 465 # 25 for public, 465 for ssl
email_username = "username"
email_passwd = "passwd"
```

### Initialization

Change Arxiv tags in `arxiv_daemon.py` and retrieve necessary paper data at the first time:

```
python arxiv_daemon.py -n XXXX -b XX -m XXXX -s XXXX
```

And compute the SVM weights:

```
python compute.py
```

### Daemons
To run this locally I usually run the following script to update the database with any new papers. I typically schedule this via an apscheduler:

```python
from apscheduler.schedulers.blocking import BlockingScheduler
import subprocess

def fetch_compute():
    subprocess.call(["python", "arxiv_daemon.py", "-n", "2000", "-m", "1000"])
    subprocess.call(["python", "compute.py"])

def send_email():
    subprocess.call(["python", "send_emails.py", "-t", "1.5"])

scheduler = BlockingScheduler(timezone="Asia/Shanghai")
scheduler.add_job(fetch_compute, "cron", day_of_week="tue,wed,thu,fri,mon", hour=14)
scheduler.add_job(
    send_email, "cron", day_of_week="tue,wed,thu,fri,mon", hour=15, minute=30
)
scheduler.start()
```

### Website
You can see that updating the database is a matter of first downloading the new papers via the arxiv api using `arxiv_daemon.py`, and then running `compute.py` to compute the tfidf features of the papers. Finally to serve the flask server locally we'd run something like (or see `up.sh`):

```bash
gunicorn -w 4 -b 0.0.0.0:5000 serve:app
```
For safety, setup your key by
```python
import secrets;
secrets.token_urlsafe(16)
```
and put it into `secret_key.txt`.

All of the database will be stored inside the `data` directory. The entire database of papers will be load into the memory for fast operation of the website. An SSD device is recommended for faster database querying.

### E-mail Notification
(Optional) Finally, if you'd like to send periodic emails to users about new papers, see the `send_emails.py` script. The primitive SMTP method is used.


## Documents

### Login
**You shold login to toggle the features!**

### Search by Words
Ranking metrics tuned for paper searching are used when you type in keywords and authors.

### Search by Tags / Pids (Papers)
An svm linear classifier will be trained to classify the papers: the papers with tags are labelled as positive.

### Keywords Tracking
Latest papers matched with the keywords added by the users will be recommended.

### Tags Tracking
Latest papers with high classification scores according to the tags and register combinations of tags will
be recommended.

### Add Tags for A Paper

![fig1](static/add_tag.png)

![fig2](static/add_tag2.png)

![fig3](static/add_tag3.png)

### Register Tag Combinations

![fig4](static/add_tag_comb.png)

![fig5](static/add_tag_comb2.png)

### License

MIT
