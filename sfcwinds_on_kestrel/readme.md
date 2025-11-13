Scripts on Kestrel for the sfcwinds project

Updates files on Github from Kestrel with:

git add .
git commit -m "Update message"
git push

Update files on Kestrel from Github with:
git pull origin main

On Kestrel, there is a sparse checkout of the sfcwinds_on_kestrel subdirectory. It clones the whole repo but only checks out a specific subdirectory (sfcwinds_on_kestrel) to the working directory on Kestrel.
