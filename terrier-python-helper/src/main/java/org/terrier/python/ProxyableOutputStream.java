package org.terrier.python;

import java.io.OutputStream;

/**
  * This class allows an OutputStream to be redirected
  * to an interface, namely OutputStreamable.
  */
public class ProxyableOutputStream extends OutputStream
{
    OutputStreamable proxy;
    public ProxyableOutputStream(OutputStreamable _proxy) {
        this.proxy = _proxy;
    }
    public void	close() { proxy.close(); }
    public void	flush() { proxy.flush(); }
    public void	write(byte[] b) { proxy.write(b); }
    public void	write(byte[] b, int off, int len) { proxy.write(b,off,len); }
    public void	write(int b) { proxy.write(b); }
}
