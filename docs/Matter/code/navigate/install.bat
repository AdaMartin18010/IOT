@echo off
rem run this script as admin

if not exist navigate.exe (
    echo Build the example before installing by running "go build"
    goto :exit
)

sc create navigate binpath= "%CD%\navigate.exe" start= auto DisplayName= "navigate"
sc description navigate "navigationlock-Service"
net start navigate
sc query navigate

echo Check navigate server's log file
 
:exit
