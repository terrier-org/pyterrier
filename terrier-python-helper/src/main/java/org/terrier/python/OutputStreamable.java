package org.terrier.python;

/** This is an interface representing an OuputStream.
  * We use an interface, as Pyjnius cannot extend Java Abstract
  * classes in Python.
  */
public interface OutputStreamable
{
    void	close();
    void	flush();
    void	write(byte[] b);
    void	write(byte[] b, int off, int len);
    void	write(int b);
}
