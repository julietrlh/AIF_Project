#!/bin/bash

# Create directories
mkdir /data
cd /data
# Download files to /data directory
wget "https://drive.google.com/uc?export=download&id=1goJZr6r8-4spnWyRHTRjwHfHTrqAB3mm" 
wget "https://drive.google.com/uc?export=download&id=1A-5umJhRa2ORMtD_Ek6mMPr63Q6ea0QT" 
wget "https://drive.google.com/uc?export=download&id=1ULaLXNkEkul1IqDaNtWi43_VRWinDp5T" 
wget "https://drive.google.com/uc?export=download&id=1EgKnD-dLk0gMYgUcURh3QuRHJ035Mbyi"
wget "https://drive.google.com/uc?export=download&id=1AgR5pa1wvfc5CbAtfHyXS-VoL6o6_VCU" 
cd ..
mkdir /model
cd /model
# Download files to /model directory
wget "https://drive.google.com/uc?export=download&id=1Rg3pr6UNvbYbTQEtufwjeag35nbkVMx6" 
wget "https://drive.google.com/uc?export=download&id=1uSybCy7jeHKn07X5uTwXVrd6xfaeHmJe" 
cd ..