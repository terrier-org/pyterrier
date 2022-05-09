# INIT001 - pt.init() has already been called. Check pt.started() before calling pt.init()

## Cause

`pt.init()` starts the Java virtual machine (JVM). This can only be done once.

## Example

The following core will raise a RuntimeError with INIT001.

```python
pt.init()

#etc

pt.init(boot_packages=['bla:bla:etc'])
```

## Resolution

You need to restart the Python process between invocations of `pt.init()`. 

To prevent re-execution (for instance when re-executing a notebook), you can use:

```python
if not pt.started():
  pt.init(boot_packages=['bla:bla:etc'])
```