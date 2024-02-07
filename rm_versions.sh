#!/bin/bash

while IFS= read -r line
do
	echo "${line%%=*}" >> requirements_no_versions.txt
done < requirements.txt

