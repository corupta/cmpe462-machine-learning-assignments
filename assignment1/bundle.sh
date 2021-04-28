#!/usr/bin/env bash

if ! [ -x "$(command -v markdown-pdf)" ]; then
  echo 'Error: markdown-pdf is not installed.'
  echo 'Please install it with npm i -g markdown-pdf';
  echo 'If npm fails, try again using node version 8';
  exit 1
fi

if ! [ -x "$(command -v ots-cli.js)" ]; then
  echo 'Error: ots-cli.js (opentimestamps-client) is not installed.'
  exit 1
fi

if ! [ -x "$(command -v zip)" ]; then
  echo 'Error: zip is not installed.'
  exit 1
fi

rm 462_assignment1_2020742015.zip
rm 462_assignment1_2020742015.zip.ots
rm 462_assignment1_2020742015_report.pdf
markdown-pdf Readme.md -o 462_assignment1_2020742015_report.pdf -r "portrait" -m "{\"html\":true,\"breaks\":false}"

zip 462_assignment1_2020742015.zip -r assignment1.py requirements.txt 462_assignment1_2020742015_report.pdf
ots-cli.js stamp 462_assignment1_2020742015.zip
ots-cli.js upgrade 462_assignment1_2020742015.zip.ots

rm -r 462_assignment1_2020742015
unzip 462_assignment1_2020742015.zip -d 462_assignment1_2020742015

# You should submit 3 items:
# ∗ your Python script, assignment1.py
# ∗ your requirements.txt file, blank file if no additional library is needed
# ∗ your assigment report in pdf, 462_assignment1_<studentid>_report.pdf, example
# 462_assignment1_20181123456_report.pdf
# – You should compress all submission items in a zip file with name as 462_assign- ment1_<studentid>.zip, example 462_assignment1_20181123456.zip
# – The zip will be submitted on Moodle.

echo "Zip 462_assignment1_2020742015.zip is created, don't forget to check its contents"
echo "Also, don't forget to push the .zip and .ots to github"
echo "Do not forget to submit the zip file on Moodle"
echo "Do not forget to submit the report to Turnitin"
echo "Or, Do not mail the report to Assistant while Instructor is on CC in the case of extension situation"
echo "Assistant e-mail: ozlem.simsek@boun.edu.tr"
echo "Instructor e-mail: inci.baytas@boun.edu.tr"
