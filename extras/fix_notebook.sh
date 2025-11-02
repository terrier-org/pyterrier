#!/bin/bash
FILE=$1

jq -M 'del(.metadata.widgets)' $FILE > $FILE.fixed
mv $FILE.fixed $FILE
