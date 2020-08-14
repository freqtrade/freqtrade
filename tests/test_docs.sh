#!/bin/bash
# Test Documentation boxes -
# !!! <TYPE>: is not allowed!
# !!! <TYPE> "title" - Title needs to be quoted!
# !!! <TYPE> Spaces at the beginning are not allowed
grep -Er '^!{3}\s\S+:|^!{3}\s\S+\s[^"]|^\s+!{3}\s\S+' docs/*

if  [ $? -ne 0 ]; then
    echo "Docs test success."
    exit 0
fi
echo "Docs test failed."
exit 1
