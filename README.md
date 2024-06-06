# Project Bayesian Statical Methods
Project for the course Bayesian Statistical Methods and Data Analysis at ETHZ 

---

## Basic Git Commands for Beginners

### Setting Up

1. **Add SSH Key to your Git Profile**:
   
    Check (on your operating system) if you already have an SSH Key. Check the following directoy:
   ```bash
   cd ~/.ssh
   ```
    If the directory does not exist or no ssh key is present, generate a new ssh key with:
    ```bash
    ssh-keygen -t rsa
    ```
    Display the public ssh key with
    ```bash
    cat ~/.ssh/id_rsa.pub
    ```
    Copy the ssh key and add it to your git ssh keys. You can find your ssh keys in the git profile settings.

    For a more comprehensive introduction see [conneting-to-github-with-ssh](https://docs.github.com/en/authentication/connecting-to-github-with-ssh).

2. **Clone a Repository**:
   If you are starting by cloning an existing repository, use:
   ```bash
   git clone <repository_url>
   ```
   Replace `<repository_url>` with the URL of the repository you want to clone. You can find the URL by pressing the green `code` button on the git repository.

### Basic Git Workflow

1. **Check Status**:
   Before making any changes, it's a good practice to check the current status of your repository:
   ```bash
   git status
   ```
   This command shows which files are changed, staged, or untracked.

2. **Add Changes**:
   After making changes to your files, you need to add them to the staging area:
   ```bash
   git add <file_name>
   ```
   To add all changed files, you can use:
   ```bash
   git add .
   ```

3. **Commit Changes**:
   Once your changes are staged, commit them with a message describing what you did:
   ```bash
   git commit -m "Your commit message"
   ```
   Replace `"Your commit message"` with a meaningful description of the changes.

4. **Push Changes**:
   To share your changes with others, push them to the remote repository:
   ```bash
   git push origin <branch_name>
   ```
   Replace `<branch_name>` with the name of the branch you are working on (e.g., `main`, `master`, `develop`).

### Additional Useful Commands

- **Create a New Branch**:
  ```bash
  git checkout -b <new_branch_name>
  ```
  Replace `<new_branch_name>` with your desired branch name.

- **Switch Branches**:
  ```bash
  git checkout <branch_name>
  ```
  Replace `<branch_name>` with the name of the branch you want to switch to.

- **Pull Changes**:
  To update your local repository with the latest changes from the remote repository:
  ```bash
  git pull origin <branch_name>
  ```
  Replace `<branch_name>` with the name of the branch you are working on.

- **View Commit History**:
  ```bash
  git log
  ```
  This command shows the commit history of your repository.



### Further Reading

- [Pro Git Book](https://git-scm.com/book/en/v2)
- [GitHub Guides](https://guides.github.com/)

---
