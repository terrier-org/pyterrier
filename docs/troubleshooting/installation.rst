Troubleshooting Installation
----------------------------

We aim to ensure that there are pre-compiled binaries available for any dependencies with native components, for all supported Python versions and for all major platforms (Linux, macOS, Windows).

If the installation failed due to ``pyautocorpus`` did not run successfully, you may need to install ``pcre`` to your machine.

macOS::

    brew install pcre

Linux::

    apt-get update -y
    apt-get install libpcre3-dev -y

For users with an M1 Mac or later models, it is sometimes necessary to install the SSL certificates to avoid certificate errors. 
To do this, locate the `Install Certificates.command` file within the `Application/Python[version]` directory. Once found, double-click on it to run the installation process.
