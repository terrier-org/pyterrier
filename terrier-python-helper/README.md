# Terrier-Python-Helper

This package contains useful Java classes for pyterrier.

## Functionalities

 - Redirecting Java stdout/stderr to the same as Python
 - Changing logging level
 - A Collection class that is useful from Java
 - Multi-threaded indexing

## Usage

Pyterrier will automatically include this package when it starts Terrier.

## Developer instructions

To install to your own Maven repo:

    mvn install

To deploy to Maven central
 
    GPG_TTY=$(tty) mvn -P release deploy

## Credits

Craig Macdonald, University of Glasgow