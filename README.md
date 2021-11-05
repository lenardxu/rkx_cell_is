rkx_cell_is
===========

This serves as a private repo for the [sartorious cell instance segmentation competition](https://www.kaggle.com/c/sartorius-cell-instance-segmentation/overview/description).

## Roadmap
- Variant 1 - modeling and training: transfer learning of state-of-the-art model trained on COCO, from on LIVECell to the 
  current dataset for the competition
  
- Variant 2 - modeling and training: transfer learning of the LIVECell-based models with adaptations on the current dataset 
  for the competition 

## Data exploration
### Data validity check
- 

## Evaluation

## Project Structure

A clear project structure helps us in collaboration and automation of tasks like automated tests.

### Folders

* `analytics`: contains all the Jupyter notebooks.
* `code`: contains python top-level package of the source code including the code of the experiments.
  * `data`: contains code for preprocessing.
  * `experiments`: contains the experiment code.
  * `model`: contains the code for modeling.
  * `utils`: contains the helper code.
* `dataset`: contains both LIVECell dataset and the dataset for competition (currently not synchronized with remote repo 
  for its size).
* `docs`: contains the references and summaries.
* `Install`: contains instruction of configuring the virtual environment based on pip3.
* `results`: contains the results of the experiments.


### Experiments Structure

Every experiment has a unique **name**.
This name is defined by the experiment and should be used to create corresponding sub folders in the global `results`.
They need to follow this structure `<NAME>/YYYY-MM-DD_hh:mm:ss/` to store results and/or figures.

A best practice is to check in only the result (e.g. as pickle) and afterwards create figures for interpretation in the notebooks.

## Contributing

### Git Workflow and Review

The collaboration requires some rules, to keep the code clean:
* We are using GitHub's pull requests model to integrate new code into our repository.
* Every pull request is reviewed by one of our maintainers.
* We foster a git history that is clean, easy to read and to review.

To guarantee this we follow the best practices described [here](https://www.git-tower.com/learn/git/ebook/en/command-line/appendix/best-practices#start).
We use a rebase workflow with short living branches for features and bug fixes (see [here](https://www.git-tower.com/learn/git/ebook/en/command-line/advanced-topics/rebase#start)).

### Git Message Format

#### Structure

* Capitalized, short (50 chars or less) summary.
* Always leave the second line blank.
* More detailed explanatory text. Wrap it to about 75 characters.

#### Text

Write your commit message in the imperative: "Fix bug" and not "Fixed bug" or "Fixes bug."
This convention matches up with commit messages generated by commands like git merge and git revert.
Further paragraphs come after blank lines.
Bullet points are okay, too.
Typically a hyphen or asterisk is used for the bullet, preceded by a single space, with blank lines in between.
Use a hanging indent for bullet points.



## Competition rules

- Submissions to this competition must be made through Notebooks.
- In order for the "Submit" button to be active after a commit, the following conditions must be met:
  - CPU Notebook <= 9 hours run-time
  - GPU Notebook <= 9 hours run-time
  - Internet access disabled
  - Freely & publicly available external data is allowed, including pre-trained models
  - Submission file must be named `submission.csv`
    
  
## Others
- [ ] Read through the LIVECell paper and code
- [ ] Perform statistical investigation into LIVECell dataset and competition dataset (which is an extended version) in 
  terms of their respective statistical property and comparison with each other, e.g., the overlapping images
  
  