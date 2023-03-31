#!/bin/bash

#clear
echo "Creating virtual environment ..."
#python3 -m venv myenv
#echo "Virtual environment created..."
#source myenv/bin/activate
#echo "Virtual environment activated..."
#python3 -m ipykernel install --user --name=myproject
#echo "New kernel created..."


#virtualenv -q -p /usr/bin/python3.5 $1
#/bin/bash $1/bin/activate
#pip install -r requirements.txt

#!/bin/bash
#service to get python packages versions
# https://pypi.org/search/?q=pylab
rm -r ./myenv
virtualenv -q -p /usr/bin/python3.5 $1
source $1/bin/activate
pip3 install -r requirements.txt
