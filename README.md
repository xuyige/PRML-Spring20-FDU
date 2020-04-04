# PRML-Sprint20-FDU

Course website for PRML Spring 2020 at Fudan University.

*The desing of this file is borrowed from [here](https://github.com/ichn-hu/PRML-Spring19-Fudan)*.

Logistics

- Instructor: Xipeng Qiu ([@xpqiu](https://github.com/xpqiu))
- Teaching Assistants: Yige Xu ([@xuyige](https://github.com/xuyige))
- Grading: 4 assignments with a total 60% weight, and a 35% final project (detailed weight maybe updated later, won't be dramatically changed), and 5% for class.

## Agreement of sharing your course work

By using this repository to submit your coursework, you agree to share your coursework with your peers during the course and publicly for following year students. If you want to keep anonymous you could create a new github account and email the teaching assistant ([ygxu18@fudan.edu.cn](mailto:ygxu18@fudan.edu.cn)) to assign you a mask student id to anonymize your submission (e.g. 23333333333 rather than 16307130177).

## Coursework Guidelines

* You should use python3 as your programming language, we would recommend you to install anaconda which ships with the required packages for you to finish your coursework.
* You should use packages like `numpy`, `matplotlib`, `pandas` or `scipy`, if not specified, do not use other non-standard packages.
* Both English and Chinese are acceptable, and there will be no difference in terms of marking as long as you can make yourself clear with your report.
* Write clearly and succinctly. Highlight your answer and key result that is asked in the coursework requirements by using **Bold** font. You should take your own risk if you intentionally or unintentionally make the marking un-straightforward.
* Bonus mark (no more than 20%) will be considered if you make more in-depth exploration or further development that could in turn inspire the coursework to be a better one and show your understanding of the course material, this should only be the case given that you have already met the requirements.
* **Do not identify yourself in your source code or report**. We will use an automated script to collect your code and report for marking in order to hide your identify, so don't spoil it by writing down your student id or name in your report. The only identification is the name of the directory that contains your submission, however it won't be revealed during marking.
* Please use the [issue system](https://github.com/xuyige/PRML-Spring20-FDU/issues) to ask questions about the coursework and course or discuss about the course content, use proper tags whenever possible, (e.g. `assignment-1`). In this case any questions answered by the instructor, TA or others, and discussions will also be valuable for other students.
* If you find any mistakes in the coursework or the course website itself (e.g. typos) you are encouraged to correct it with a pull request, however, don't mix this kind of changes with your coursework submission pull request as stated in the following section.
* For any feedback, please consider emailing the TA first.


## Submission Guidelines

We assume that you are familiar with GitHub and git in general, if not please search online and ask your friends for help, although we will give you some hints bellow.

For each assignment, the file should be structured like this


```
.
├── assignment-1/
│   ├── 16307130177/
│       ├── report.pdf
│       └── source.py
|       └── handout/
|           └── __init__.py
```

The `handout/` directory contains the facilities provided for you to accomplish the assignment, they will most likely be provided as python functions so that you could import it to your source code by adding the `..` to your python path (see [here](assignment-1/example/source.py) for example).


The workflow of doing the coursework is like:

1. You fork this repository, and clone your forked repository to your local workplace.  
  ```
  git clone git@github.com:your_username/PRML-Spring20-FDU.git
  ```

2. When new assignment is released, first pull the updates in the original repository in your local cloned workplace, and create a directory with your student id (or mask student id) under the assignment directory, and only work in your own directory so that you won't conflict with others.  
  ```
  # First, open a terminal and navigate to the root of your local repository. Then type:
  git pull git@github.com:xuyige/PRML-Spring20-FDU.git
  # this command will update your local repository with the upstream update for the coursework
  # then make a directory for your submission
  cd assignment-1 # assignment-2 or so
  mkdir 16307130177 # create your working directory, substitute 16307130177 with your student id
  cd 16307130177 # then make sure all your work is done in this directory
  ```

3. Once you have finished your work and are ready for submission, you could use the following command to submit it (note that your are also allowed to commit your intermediate result as a means of preserve and track your progress, however you have to make sure that 1) you don't commit useless files, 2) you properly name your commit)
  ```
  git add . # add your submission into git
  git commit -m "submission of assignment-1 of 16307130177" # identify your submission
  git push # push your local submission to your remote forked repository
  ```  
  Once you've finished this, you need to open your browser to the page of your forked repository and create a pull request for your submission, we will update a simple tutorial with illustrations later.  
  **The time your create a pull request will be seen as the time you submit your coursework**, so make sure you do this before the due date.  
  As stated in the above section, you are welcomed to enhance the course material by another pull request, however, you **should not** mix these kind of commit with your coursework commit, because your enhancement pull request might be rejected, however we don't want to reject your coursework pull request, so please be extremely aware of this.

## Notice

- In order to make the repository clean and easy for marking, you are only allowed to commit `*.py`, `report.pdf`, and `*.data` (if required, no more than 20Kb) files for your submission (take a look at the `.gitignore` file for details).
- You could organize your code with hierarchy files later when the coursework becomes more complicated, however you should make your main procedure in `source.py` file.  
- Note that you could use `jupyter notebook` for your exploration and experiment, however when you want to submit your coursework, please re-organize your code into plain python file. Also note that your report should be a `pdf` file named `report.pdf` and you have to make it as small as possible, since we have 30+ students and 4 assignments.
- For external data in a large scale, please note the link to the data mentioned in your report.
- In your report, please note the command lines for running your source code and make sure that your source code can be run correctly.
- Note that pull request after the deadline is *unavailable*. For each assignment, you can push your source code to the main repository for **no more than 2 times**.
- Note that your source code should be as clean as possible. Useless code should not be included.
