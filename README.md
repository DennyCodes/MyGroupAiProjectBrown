# WHAT TO DO:
- Finish optimizer and get accuracy percentages 
- Get API working with new version of code
- PRIORITY: Get Demo working (send audio in, get who said it out)
    - Training data (recording sets of phrases under PHRASES.md)
        - Record on your own and just send it to me under gabrieliervolino1802@gmail.com

# Basic git workflow:
#### - Clone the repo
```
git clone https://github.com/gabiervo/voice_recog_final_project.git
```

#### - Edit files and then add them with:
```
git add .
```
The '.' just adds every file, if you only want to add a specific file just indicate its name
```
git add filename.file_extension
```

##### - Check that everything is working
```
git status
```
This should show the files you added in 'git add'

#### - Commit it to the repo (this does not upload the files yet)
```
git commit -m "description of what you did"
```



#### - Push it to the main repository:
```
git push
```




## Working with branches:
To avoid pushing bad code to the main branch we will use separate branches for everything until the code is proven to work:

#### Create a branch:
```
git branch branch_name
```

#### Check available branches:
```
git branch
```

#### Move to one branch or another:
```
git checkout branch_name
```

#### How to merge branches to main (while being on the main branch):
```
git merge branch_name
```
### NOTE: this does not delete the branch that was merged with, to delete use:
```
git branch -d branch_name
```
